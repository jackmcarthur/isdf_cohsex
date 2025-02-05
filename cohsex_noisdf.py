import numpy as np
import cupy as cp
from wfnreader import WFNReader
import fftx
import symmetry_maps
import matplotlib.pyplot as plt
if cp.cuda.is_available():
    xp = cp
else:
    xp = np

# return ranges of bands necessary for \sigma_{X,SX,COH}
def get_bandranges(nv,nc,nband,nelec):
    """Return ranges of bands necessary for \sigma_{X,SX,COH}"""
    nvrange = [int(nelec-nv), int(nelec)]
    ncrange = [int(nelec), int(nelec+nc)]
    nsigmarange = [int(nelec-nv), int(nelec+nc)]
    n_fullrange = [0, int(nband)]
    n_valrange = [0, int(nelec)]
    return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange

def wrap_points_to_voronoi(randcart, bvec,xp, nmax=1):
    """
    Helper function to get test q-points for mini-BZ average with correct voronoi cell.
    """
    # 1. Generate all candidate integer translations.
    grid = xp.arange(-nmax, nmax+1)
    # meshgrid in 3D; shape will be (3, M) with M = (2*nmax+1)**3 candidates.
    shifts = xp.stack(xp.meshgrid(grid, grid, grid, indexing='ij'), axis=-1).reshape(-1, 3)
    
    # 2. Convert integer translations into Cartesian shift vectors.
    #    Here bvec.T has lattice vectors as rows.
    candidate_shifts = shifts @ bvec  # shape (M, 3)

    # 3. For each point, compute its distance to each candidate image.
    #    randcart[:, None, :] has shape (N,1,3), candidate_shifts[None,:,:] has shape (1,M,3)
    diff = randcart[:, None, :] - candidate_shifts[None, :, :]  # shape (N, M, 3)
    dists = xp.linalg.norm(diff, axis=2)  # shape (N, M)

    # 4. Select, for each point, the candidate that minimizes the distance.
    best_idx = xp.argmin(dists, axis=1)  # shape (N,)
    best_shifts = candidate_shifts[best_idx]  # shape (N, 3)

    # 5. Wrap the points by subtracting the chosen lattice translation.
    wrapped = randcart - best_shifts
    return wrapped


def get_V_qG(wfn, sym, q0, xp, sys_dim):
    # first: V(q,G,G') = 4\pi/|q+G|^2 \delta_{G,G'} * trunc part in 2D, (1-exp(-zc*kxy)*cos(kz*zc))
    #print("vqg start")
    bvec = xp.asarray(wfn.blat * wfn.bvec, dtype=xp.float64)
    q0xp = xp.asarray(q0, dtype=xp.float64)
    qvec = xp.array([xp.float64(0.),xp.float64(0.),xp.float64(0.)])
    #print("vqg qvec done")
    G_q_crys = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    G_cart = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    #print("vqg G_q_crys done")
    V_qG = xp.zeros((sym.nk_tot, int(wfn.ngkmax)), dtype=xp.float64)
    ngks = xp.asarray(wfn.ngk, dtype=xp.int32)
    #print("vqg all arrays done")

    # if sys dim == 3 return error not implemented
    if sys_dim == 3:
        raise NotImplementedError("3D system calculation not yet implemented")
    #print('trying vqg')
    # get V(q,G) array for all sym-reduced q
    if sys_dim == 2:
        for iq in range(sym.nk_tot):
            qvec = xp.asarray(sym.unfolded_kpts[iq])
            print(qvec.shape)
            if iq == 0:
                qvec = q0xp
            Gmax_q = ngks[sym.irk_to_k_map[iq]]

            G_q_crys.fill(0.)
            G_cart.fill(0.)
            # this saves memory in the case of many kpts but requires a lot of HtoD transfers. revisit.
            G_q_crys[:Gmax_q] = xp.asarray(sym.get_gvecs_kfull(wfn,iq).astype(np.float64),dtype=xp.float64) # stored as int32, trying to convert efficiently
            G_cart[:Gmax_q] = xp.matmul(G_q_crys[:Gmax_q] + qvec, bvec) # @ is super slow, probably using numpy
            #print("done with gcart")
            V_qG[iq,:Gmax_q] = xp.divide(4*xp.pi, xp.sum(G_cart*G_cart, axis=1)[:Gmax_q])
            #print("done with vqg no trunc")
            kxy = xp.linalg.norm(G_cart[:Gmax_q,:2], axis=1)
            kz = G_cart[:Gmax_q,2]
            zc = xp.pi/bvec[2,2] # note that the crystal z axis must align with the cartesian z axis
            # NOT SURE WHY THERES AN EXTRA 2. 8PI NOT 4PI? I\neq J probably? but i compared to an epsmat.h5 file
            V_qG[iq,:Gmax_q] *= 2 * (1-xp.exp(-zc*kxy)*xp.cos(kz*zc))

        # mini-BZ voronoi monte carlo integration for V_q=0,G=0
        randlims = xp.matmul(bvec.T, xp.matmul(xp.diag(xp.divide(1.0, xp.asarray(wfn.kgrid))), xp.linalg.inv(bvec.T)))

        # BGW VORONOI CELL AVERAGE
        randvals = xp.random.rand(1500000,3)
        randcart = xp.einsum('ik,jk->ji', bvec.T, randvals)
        wrapped_cart = wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)

        randqcart = xp.einsum('ik,jk->ji', randlims, wrapped_cart) # set of non-grid qpts closer to q=0 than any other qpt
        #randqcart = xp.einsum('ik,jk->ji', bvec.T, randqs)
        # DEBUG: possibly necessary in 2d?
        randqcart[:,2] = 0.0
        rand_vq = xp.divide(4*xp.pi, xp.einsum('ij,ij->i',randqcart,randqcart))


        rand_vq *= 2 * (1. - xp.exp(-xp.pi/bvec[2,2] * xp.linalg.norm(randqcart[:,:2], axis=1)) * xp.cos(randqcart[:,2] * xp.pi/bvec[2,2]))

        #print(f"V_q=0,G=0 from q0: {V_qG[0,0]:.4f}")
        V_qG[0,0] = xp.mean(rand_vq)
        print(f"V_q=0,G=0 from miniBZ monte carlo: {V_qG[0,0]:.4f}")

    return V_qG


def fft_bandrange(wfn, sym, bandrange, is_left, psi_rtot_out, xp=cp):
    """Process wavefunctions for all k-points in the full Brillouin zone.
    Args:
        wfn/sym: WFNReader/SymMaps objects
        bandrange: Tuple (start, end) for band range
        is_left: Bool indicating if psi = psi_l (gets conjugated)
    Returns:
        psi_rtot_out: Array of real-space wavefunctions for all k-points
    """
    # Get dimensions
    nb = bandrange[1] - bandrange[0]
    n_rtot = int(xp.prod(wfn.fft_grid))
    nspinor = wfn.nspinor

    # Initialize temporary arrays
    psi_rtot = xp.zeros((nb, nspinor, *wfn.fft_grid), dtype=xp.complex128)
    
    # Loop over all k-points in full BZ
    for k_idx in range(sym.nk_tot):
        # Get reduced k-point index and symmetry operation
        # note these both take the unfolded k-point index
        k_red = sym.irk_to_k_map[k_idx]
        # Initialize G-space wavefunctions
        psi_Gspace = xp.zeros((nb, nspinor, wfn.ngk[k_red]), dtype=xp.complex128)
        
        # Get G-vectors and rotate them
        gvecs_k_rot = xp.asarray(sym.get_gvecs_kfull(wfn,k_idx))
        # Get wavefunction coefficients (symmetry unfolded)
        for ib, band_idx in enumerate(range(bandrange[0], bandrange[1])):
            psi_Gspace[ib, :, :] = xp.asarray(sym.get_cnk_fullzone(wfn,band_idx,k_idx))
        
        # FFT to real space
        psi_rtot.fill(0.+0.j)
        for ib in range(nb):
            for ispinor in range(nspinor):
                # Place G-space coefficients
                psi_rtot[ib,ispinor,gvecs_k_rot[:,0],gvecs_k_rot[:,1],gvecs_k_rot[:,2]] = psi_Gspace[ib,ispinor,:]
                # Perform FFT
                psi_rtot[ib,ispinor] = fftx.fft.fftn(psi_rtot[ib,ispinor])
                # CHANGING FROM IFFTN, SEE BGW/SIGMA/MTXEL.F90
        
        # Normalize
        psi_rtot *= xp.sqrt(1./n_rtot)
        
        # Store results with new ordering
        if is_left:
            psi_rtot_out[k_idx] = xp.conj(psi_rtot)
        else:
            psi_rtot_out[k_idx] = psi_rtot


def get_sigma_x_exact(wfn, sym, k_r, bandrange_l, bandrange_r,V_qG,xp):
    """Get the bare exchange self-energy, Sigma_X, for the 0th band in bandrange_r, for valbands = bandrange_l"""
    # Get dimensions
    n_rtot = int(np.prod(wfn.fft_grid))
    nb_l = bandrange_l[1] - bandrange_l[0]
    nb_r = bandrange_r[1] - bandrange_r[0]
    nspinor = wfn.nspinor
    
    # Initialize output arrays that hold all relevant u_nk(r)with (nk, nb) ordering
    psi_l_rtot_out = xp.zeros((sym.nk_tot, nb_l, nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot_out = xp.zeros((sym.nk_tot, nb_r, nspinor, *wfn.fft_grid), dtype=xp.complex128)

    # Initialize temporary arrays for processing
    # notice combined band/spinor index, so we can use a single cublas matmul call later
    psi_l_rtot = xp.zeros((nb_l*nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot = xp.zeros((nb_r*nspinor, *wfn.fft_grid), dtype=xp.complex128)


    # Initialize exp(iGr) phase factor arrays outside kq loops
    fft_nx, fft_ny, fft_nz = wfn.fft_grid
    fx = xp.arange(fft_nx, dtype=float)[None, :, None, None] / fft_nx  # Shape: (nx,1,1)
    fy = xp.arange(fft_ny, dtype=float)[None, None, :, None] / fft_ny  # Shape: (1,ny,1)
    fz = xp.arange(fft_nz, dtype=float)[None, None, None, :] / fft_nz  # Shape: (1,1,nz)

    # Pre-allocate phase arrays
    px = xp.zeros((1,fft_nx, 1, 1), dtype=xp.complex128)
    py = xp.zeros((1,1, fft_ny, 1), dtype=xp.complex128)
    pz = xp.zeros((1,1, 1, fft_nz), dtype=xp.complex128)

    # initialize array that holds M_mn(k,-q,-G) in reciprocal space
    psiprod_flat = xp.zeros(int(wfn.ngkmax), dtype=xp.complex128)

    # fill psi_l/r_rtot_out with respective psi(*)_l/r(r) for all k
    fft_bandrange(wfn, sym, bandrange_l, True, psi_l_rtot_out, xp=cp)
    fft_bandrange(wfn, sym, bandrange_r, False, psi_r_rtot_out, xp=cp)

    psi_r_rtot[:] = psi_r_rtot_out[k_r].reshape(nb_r*2,*wfn.fft_grid)
    sigma_out = 0.0+0.0j
    psiprod = xp.zeros(wfn.fft_grid, dtype=xp.complex128)
    
    ##########################################
    # Loop over all (unfolded) (k-q)-points (no symmetries used for q here)
    # for each q, get M_(vrange k-q,nk)(G)
    ##########################################
    for k_l in range(int(sym.nk_tot)):
        # qvec in extended zone (= k-(k-q))
        q_ext = xp.asarray(sym.unfolded_kpts[k_r] - sym.unfolded_kpts[k_l])
        q_rounded = xp.round(q_ext)
        q_ext = xp.where(xp.abs(q_ext - q_rounded) < 1e-8, q_rounded, q_ext)
        # convention used in Deslippe 2012: qbz = qS + G_q (no syms for now)
        G_q = xp.asarray(q_ext%1.0 - q_ext, dtype=xp.int32)
        iq = find_qpoint_index(q_ext, sym, tol=1e-6)
        iq_cpu = iq.get()

        # the G components associated with the q point are shifted because V_q is stored in first BZ
        G_q_comps = xp.asarray(sym.get_gvecs_kfull(wfn,iq_cpu), dtype=xp.int32) + G_q

        psi_l_rtot = psi_l_rtot.reshape(nb_l,nspinor,*wfn.fft_grid)
        psi_r_rtot = psi_r_rtot.reshape(nb_r,nspinor,*wfn.fft_grid)
            
        psi_l_rtot[:] = psi_l_rtot_out[k_l].reshape(nb_l,2,*wfn.fft_grid)

            ##############################################
            # phase factor for psi_l = psi_nk-q:
        k_l_full = xp.asarray(sym.unfolded_kpts[k_l] - sym.unfolded_kpts[iq_cpu])
        rounded = xp.round(k_l_full)
        k_l_full = xp.where(xp.abs(k_l_full - rounded) < 1e-8, rounded, k_l_full)

        # here we get:
        # M_vn(k,-q,-G) = \sum_a FFT[u_vk-q,a(r) u_nk,a(r)]
        # <nk|Sigma_X|nk> = \sum_G M_vn(k,-q,-G)^* V_q(G) M_vn(k,-q,-G)
        for ib in range(psi_l_rtot.shape[0]):
            psiprod_flat[:] = 0.0+0.0j
            for ispinor in range(2):
                psiprod = psi_l_rtot[ib,ispinor] * psi_r_rtot[0,ispinor]
                psiprod = fftx.fft.fftn(psiprod)
                psiprod *= xp.sqrt(1./xp.float64(n_rtot))

                psiprod_flat[:G_q_comps.shape[0]] += psiprod[-G_q_comps[:,0],-G_q_comps[:,1],-G_q_comps[:,2]]


            sigma_out += xp.sum(xp.conj(psiprod_flat) * V_qG[iq] * psiprod_flat)

    return -sigma_out


def find_qpoint_index(q_ext, sym, tol=1e-6):
    """Find index of q-point in unfolded k-points list.
    
    Args:
        q_ext: Vector of length 3 (crystal coordinates)
        sym: SymMaps object containing unfolded_kpts
        tol: Tolerance for floating point comparison
    
    Returns:
        Index of matching q-point, or raises ValueError if not found
    """
    # Get fractional part of q_ext
    q_frac = q_ext % 1.0
    
    # Calculate differences with all unfolded k-points
    diffs = xp.abs(xp.asarray(sym.unfolded_kpts) - q_frac[None, :])
    # Sum over coordinates and find minimum difference
    total_diffs = xp.sum(diffs, axis=1)
    min_diff = xp.min(total_diffs)
    
    if min_diff > tol:
        raise ValueError(f"No matching q-point found within tolerance {tol}")
    
    return xp.argmin(total_diffs)




if __name__ == "__main__":
    nval = 5
    ncond = 5
    nband = 30

    sys_dim = 2 # 3 for 3D, 2 for 2D

    wfn = WFNReader("WFN.h5")
    wfnq = WFNReader("WFNq.h5")
    sym = symmetry_maps.SymMaps(wfn)
    q0 = wfnq.kpoints[0] - wfn.kpoints[0]
    if np.linalg.norm(q0) > 1e-6:
        print(f'Using q0 = ({q0[0]:.5f}, {q0[1]:.5f}, {q0[2]:.5f})')

    nvrange, ncrange, nsigmarange, n_fullrange, n_valrange = get_bandranges(nval, ncond, nband, wfn.nelec)

    ####################################
    # 1.) get (truncated in 2D) coulomb potential v_q(G)
    ####################################
    V_qG = get_V_qG(wfn, sym, q0, xp, sys_dim)
    
    #for i in range(21,31):
    #    sigma = get_sigma_x_exact(wfn, sym, 0, n_valrange, (i,i+1), V_qG, xp)
    #    fact = 1.0/(wfn.cell_volume*sym.nk_tot)
    #    print(f"{sigma.real:.5f}")# + {sigma.imag:.5f}j")


