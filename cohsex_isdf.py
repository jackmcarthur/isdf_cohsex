import numpy as np
import cupy as cp
from wfnreader import WFNReader
from epsreader import EPSReader
import fftx
import symmetry_maps
import cupyx.scipy.fftpack
#import matplotlib.pyplot as plt
if cp.cuda.is_available():
    xp = cp
else:
    xp = np

# return ranges of bands necessary for \sigma_{X,SX,COH}
def get_bandranges(nv,nc,nband,nelec):
    """Return ranges of bands necessary for \\sigma_{X,SX,COH}"""
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

def get_V_qG(wfn, sym, q0, xp, epshead, sys_dim):
    # first: V(q,G,G') = 4\pi/|q+G|^2 \delta_{G,G'} * trunc part in 2D, (1-exp(-zc*kxy)*cos(kz*zc))
    # (times one other factor, 1/(N_ktot * cell_volume))
    #print("vqg start")
    print(q0)
    bvec = xp.asarray(wfn.blat * wfn.bvec, dtype=xp.float64)
    q0xp = xp.asarray(q0, dtype=xp.float64)
    qvec = xp.array([xp.float64(0.),xp.float64(0.),xp.float64(0.)])
    zc = xp.pi/bvec[2,2] # note that the crystal z axis must align with the cartesian z axis

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
        for iq in range(wfn.nkpts):
            qvec = xp.asarray(wfn.kpoints[iq])
            print(qvec.shape)
            if iq == 0:
                qvec = q0xp

            Gmax_q = ngks[iq]

            G_q_crys.fill(0.)
            G_cart.fill(0.)
            # this saves memory in the case of many kpts but requires a lot of HtoD transfers. revisit.
            G_q_crys[:Gmax_q] = xp.asarray(wfn.get_gvec_nk(iq).astype(np.float64),dtype=xp.float64) # stored as int32, trying to convert efficiently
            G_cart[:Gmax_q] = xp.matmul(G_q_crys[:Gmax_q] + qvec, bvec) # @ is super slow, probably using numpy

            #print("done with gcart")
            V_qG[iq,:Gmax_q] = xp.divide(4*xp.pi, xp.sum(G_cart*G_cart, axis=1)[:Gmax_q])
            #print("done with vqg no trunc")
            kxy = xp.linalg.norm(G_cart[:Gmax_q,:2], axis=1)
            kz = G_cart[:Gmax_q,2]
            # NOT SURE WHY THERES AN EXTRA 2. 8PI NOT 4PI? I\neq J probably? but i compared to an epsmat.h5 file
            V_qG[iq,:Gmax_q] *= 2 * (1-xp.exp(-zc*kxy)*xp.cos(kz*zc))

        # mini-BZ voronoi monte carlo integration for V_q=0,G=0
        randlims = xp.matmul(bvec.T, xp.matmul(xp.diag(xp.divide(1.0, xp.asarray(wfn.kgrid))), xp.linalg.inv(bvec.T)))

        # BGW VORONOI CELL AVERAGE
        randvals = xp.random.rand(2500000,3)
        randcart = xp.einsum('ik,jk->ji', bvec.T, randvals)
        wrapped_cart = wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)

        randqcart = xp.einsum('ik,jk->ji', randlims, wrapped_cart) # set of non-grid qpts closer to q=0 than any other qpt
        #randqcart = xp.einsum('ik,jk->ji', bvec.T, randqs)
        # DEBUG: possibly necessary in 2d?
        randqcart[:,2] = 0.0
        rand_vq = xp.divide(4*xp.pi, xp.einsum('ij,ij->i',randqcart,randqcart))
        kxy_q0 = xp.linalg.norm(randqcart[:,:2],axis=1)
        rand_vq *= 2 * (1. - xp.exp(-xp.pi/bvec[2,2] * kxy_q0) * xp.cos(randqcart[:,2] * xp.pi/bvec[2,2]))
        #print(f"V_q=0,G=0 from q0: {V_qG[0,0]:.4f}")
        V_qG[0,0] = xp.mean(rand_vq)
        print(f"V_q=0,G=0 from miniBZ monte carlo: {V_qG[0,0]:.4f}")

        ##############################################################
        # this is wcoul0 used in BGW/Common/fixwing.f90 (generated in minibzaverage.f90)
        # equations here are: (Ismail-Beigi PRB 2006)
        # W(q,G=G'=0) = epsinv(q,G=G'=0) * vc(q)
        # 1/epsinv(q,G=G'=0) = 1 + vc(q)*f(q)
        # f(q) = gamma |q|^2 exp(-a|q|) (a=0 in minibzaverage.f90)

        q0len = xp.linalg.norm(xp.matmul(q0xp, bvec))
        vc_qtozero = (1.-xp.exp(-q0len*zc))/q0len**2
        gamma = xp.float64((1./xp.asarray(epshead.real, dtype=xp.float64) - 1.)/(q0len**2 * vc_qtozero))
        alpha = xp.float64(0.)

        rand_wq = (1. - xp.exp(-kxy_q0*zc))/(kxy_q0**2) # actually vc(q)
        rand_wq = xp.divide(rand_wq, (1. + rand_wq * kxy_q0**2 * gamma *xp.exp(-alpha*kxy_q0)))
        wcoul0 = 8*xp.pi*xp.mean(rand_wq)

        print(f"W_q=0(G=G'=0) from miniBZ monte carlo: {wcoul0:.4f}")

        fact = xp.float64(1./(sym.nk_tot*wfn.cell_volume)) # won't work if nonuniform grid
        V_qG *= fact
        wcoul0 *= fact
    return V_qG.astype(xp.complex128), wcoul0.astype(xp.complex128)


def fft_bandrange(wfn, sym, bandrange, is_left, psi_rtot_out, xp=cp):
    """
    Get psi_nk(r) for all k-points in the full Brillouin zone.
    (not u_nk(r)! returns psi_nk(r) = e^{ikr} u_nk(r))
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

    # Initialize exp(ikr) phase factor arrays 
    fft_nx, fft_ny, fft_nz = wfn.fft_grid
    fx = xp.arange(fft_nx, dtype=float)[None, :, None, None] / fft_nx  # Shape: (nx,1,1)
    fy = xp.arange(fft_ny, dtype=float)[None, None, :, None] / fft_ny  # Shape: (1,ny,1)
    fz = xp.arange(fft_nz, dtype=float)[None, None, None, :] / fft_nz  # Shape: (1,1,nz)

    # Pre-allocate phase arrays
    px = xp.zeros((1,fft_nx, 1, 1), dtype=xp.complex128)
    py = xp.zeros((1,1, fft_ny, 1), dtype=xp.complex128)
    pz = xp.zeros((1,1, 1, fft_nz), dtype=xp.complex128)
    
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
                psi_rtot[ib,ispinor] = fftx.fft.ifftn(psi_rtot[ib,ispinor])

        # multiply by exp(ikr) phase factor
        k_gpu = xp.asarray(sym.unfolded_kpts[k_idx], dtype=xp.float64)
        xp.exp(2j * xp.pi * k_gpu[0] * fx, out=px)
        xp.exp(2j * xp.pi * k_gpu[1] * fy, out=py)
        xp.exp(2j * xp.pi * k_gpu[2] * fz, out=pz)
        psi_rtot *= px
        psi_rtot *= py
        psi_rtot *= pz
        
        # Store results with new ordering
        if is_left:
            psi_rtot_out[k_idx] = xp.conj(psi_rtot)
        else:
            psi_rtot_out[k_idx] = psi_rtot


def get_zeta_q_and_v_q_mu_nu(wfn, wfnq, sym, centroid_indices, bandrange_l, bandrange_r,V_qG,xp):
    """Find the interpolative separable density fitting representation."""
    # Get dimensions
    n_rtot = int(np.prod(wfn.fft_grid))
    n_rmu = int(centroid_indices.shape[0])
    nb_l = bandrange_l[1] - bandrange_l[0]
    nb_r = bandrange_r[1] - bandrange_r[0]
    nspinor = wfn.nspinor
    kgridgpu = xp.asarray(wfn.kgrid, dtype=xp.int32)

    # Initialize output arrays with (nk, nb) ordering
    psi_l_rtot_out = xp.zeros((sym.nk_tot, nb_l, nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot_out = xp.zeros((sym.nk_tot, nb_r, nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_l_rmu_out = xp.zeros((sym.nk_tot, nb_l, nspinor*n_rmu), dtype=xp.complex128)
    psi_r_rmu_out = xp.zeros((sym.nk_tot, nb_r, nspinor*n_rmu), dtype=xp.complex128)
    
    # Initialize temporary arrays for processing (1 kpt, bands in bandrange) at a time
    # notice combined band/spinor index, so we can use a single cublas matmul call later
    psi_l_rtot = xp.zeros((nb_l*nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot = xp.zeros((nb_r*nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_l_rmu = xp.zeros((nb_l*nspinor, n_rmu), dtype=xp.complex128)
    psi_r_rmu = xp.zeros((nb_r*nspinor, n_rmu), dtype=xp.complex128)
    psi_l_rmuT = xp.zeros((n_rmu, nb_l*nspinor), dtype=xp.complex128)
    psi_r_rmuT = xp.zeros((n_rmu, nb_r*nspinor), dtype=xp.complex128)

    # Initialize ZC^T and CC^T
    ZCT = xp.zeros((n_rmu, n_rtot), dtype=xp.complex128)
    CCT = xp.zeros((n_rmu, n_rmu), dtype=xp.complex128)

    # note zeta_q only stores one q at a time.
    zeta_q = xp.zeros((n_rmu, n_rtot), dtype=xp.complex128)
    zeta_qG_mu = xp.zeros((n_rmu, int(wfn.ngkmax)), dtype=xp.complex128)

    # Initialize exp(iqr) phase factor arrays outside kq loops
    fft_nx, fft_ny, fft_nz = wfn.fft_grid
    fx = xp.arange(fft_nx, dtype=float)[None, :, None, None] / fft_nx  # Shape: (nx,1,1)
    fy = xp.arange(fft_ny, dtype=float)[None, None, :, None] / fft_ny  # Shape: (1,ny,1)
    fz = xp.arange(fft_nz, dtype=float)[None, None, None, :] / fft_nz  # Shape: (1,1,nz)

    # Pre-allocate phase arrays
    px = xp.zeros((1,fft_nx, 1, 1), dtype=xp.complex128)
    py = xp.zeros((1,1, fft_ny, 1), dtype=xp.complex128)
    pz = xp.zeros((1,1, 1, fft_nz), dtype=xp.complex128)

    # initialize output V_q,mu,nu array
    V_qfullG = xp.zeros((int(wfn.ngkmax)), dtype=xp.complex128)
    V_qmunu = xp.zeros((*wfn.kgrid, n_rmu, n_rmu), dtype=xp.complex128)

    # fill psi_l/r_rtot_out with respective psi(*)_l/r(r) for all k
    print(f"Performing FFTs for wavefunction ranges {bandrange_l} and {bandrange_r}")
    fft_bandrange(wfn, sym, bandrange_l, True, psi_l_rtot_out, xp=cp)
    fft_bandrange(wfn, sym, bandrange_r, False, psi_r_rtot_out, xp=cp)
    print("FFTs complete")
    
    ##########################################
    # Loop over all q-points
    # for each q, we get zeta_q via least squares, then immediately get W/v_q,mu,nu so we don't have to store all zeta_q
    ##########################################
    qvec = xp.array([0,0,0], dtype=xp.int32)
    for qvec_nonneg in xp.ndindex(*wfn.kgrid): 
        # where qvec[i] > ceil(kgrid[i]/2), umklapp to negative (for k<->R FFTs later)
        # note: not necessary
        qvec = xp.asarray(qvec_nonneg)
        if qvec_nonneg[0] > kgridgpu[0]//2: qvec[0] = qvec_nonneg[0] - kgridgpu[0]
        if qvec_nonneg[1] > kgridgpu[1]//2: qvec[1] = qvec_nonneg[1] - kgridgpu[1]
        if qvec_nonneg[2] > kgridgpu[2]//2: qvec[2] = qvec_nonneg[2] - kgridgpu[2]

        ZCT.fill(0)
        CCT.fill(0)

        # note no symmetries used here but it would be possible to do so, though it's not a bottleneck.
        for k_r in range(sym.nk_tot):
            k_l_full = xp.asarray(sym.kvecs_asints[k_r]) - qvec#.get()
            k_l_gt0 = xp.mod(k_l_full, kgridgpu)
            k_l = np.where(np.all(sym.kvecs_asints == k_l_gt0.get(), axis=1))[0][0]


            psi_l_rtot = psi_l_rtot.reshape(nb_l*nspinor,*wfn.fft_grid)
            psi_r_rtot = psi_r_rtot.reshape(nb_r*nspinor,*wfn.fft_grid)
            
            psi_l_rtot[:] = psi_l_rtot_out[k_l].reshape(nb_l*nspinor,*wfn.fft_grid)
            psi_r_rtot[:] = psi_r_rtot_out[k_r].reshape(nb_r*nspinor,*wfn.fft_grid)

            ##############################################

            psi_l_rmu = psi_l_rtot[:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]#.reshape(nb_l*2, -1)
            psi_r_rmu = psi_r_rtot[:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]#.reshape(nb_r*2, -1)
            psi_l_rmuT = xp.ascontiguousarray(psi_l_rmu.T) # memory locality in matmuls. extra mem may be a negative
            psi_r_rmuT = xp.ascontiguousarray(psi_r_rmu.T)

            psi_l_rtot = psi_l_rtot.reshape(nb_l*nspinor, -1)
            psi_r_rtot = psi_r_rtot.reshape(nb_r*nspinor, -1)
            
            # Add contribution from this k,q pair to ZC^T and CC^T
            # combine band and spinor indices, so that decomp is of form \sum_\mu \zeta^q_\mu(r) \psi^*_mk-q,a(r_\mu) \psi_nk,b(r_\mu)
            # a.k.a. all four possible M_sigma,sigma' combinations are included
            P_l = xp.matmul(psi_l_rmuT, psi_l_rtot)
            P_r = xp.matmul(psi_r_rmuT, psi_r_rtot)
            ZCT += xp.multiply(P_l, P_r)
            
            Pmu_l = xp.matmul(psi_l_rmuT, psi_l_rmu)
            Pmu_r = xp.matmul(psi_r_rmuT, psi_r_rmu)
            CCT += xp.multiply(Pmu_l, Pmu_r)
        

        # Solve for zeta_q
        # C_C_T shape (nrmu, nrmu), Z_C_T shape (nrmu, nrtot)
        # zeta_q shape (nrmu, nrtot) 
        zeta_q = xp.linalg.lstsq(CCT, ZCT, rcond=-1)[0]
        
        # fourier transform zeta_q to G space
        zeta_q = zeta_q.reshape(n_rmu, *wfn.fft_grid)
        # remove phase factor from zeta_q(r) = e^{iqr} z_q(r)
        # qvec chosen to be potentially negative but it really shouldnt matter if consistent with vqG
        xp.exp(-2j * xp.pi * qvec[0] / kgridgpu[0] * fx, out=px)
        xp.exp(-2j * xp.pi * qvec[1] / kgridgpu[1] * fy, out=py)
        xp.exp(-2j * xp.pi * qvec[2] / kgridgpu[2] * fz, out=pz)
        zeta_q *= px
        zeta_q *= py
        zeta_q *= pz

        for mu in xp.ndindex(zeta_q.shape[0]):
            zeta_q[mu] = fftx.fft.fftn(zeta_q[mu])
            # this should maybe be an ifftn.
        zeta_q *= n_rtot #xp.sqrt(1./xp.float64(n_rtot))  # normalize FFT

        #####################################
        # now, get this V_qG from the stored V_qbarG array by remapping G components.
        #####################################
        # get qbar_idx, Sq and G_Sq such that q_ext = Sq @ q_ext + G_Sq.
        qveccrys = xp.divide(xp.asarray(qvec, dtype=xp.float64), kgridgpu)
        q_rounded = xp.round(qveccrys)
        q_ext = xp.where(xp.abs(qveccrys - q_rounded) < 1e-8, q_rounded, qveccrys)
        # iq is the qvec index in sym.unfolded_kpts, but iqbar/iqbareps will be the irrBZ indices used to get vcoul/epsinv. 
        iq = find_qpoint_index(q_ext, sym, tol=1e-6) 
        iq_cpu = iq.get()

        ######################################################
        # IF W_q/V_q SYMMETRY REDUCED:
        ######################################################
        # get qbar_idx, Sq and G_Sq such that q_ext = Sq @ q_ext + G_Sq.
        iqbar = sym.irk_to_k_map[iq_cpu]
        Sq = sym.sym_mats_k[sym.irk_sym_map[iq_cpu]] # now, qbar @ Sq = q
        #G_Sq = np.round(sym.unfolded_kpts[iq_cpu] - Sq @ wfn.kpoints[iqbar]).astype(np.int32)
        G_Sq = np.round(q_ext.get() - Sq @ wfn.kpoints[iqbar]).astype(np.int32) # if we needed q outside zone, but it seems fine
        vcoul_psiG_comps = xp.asarray(np.einsum('ij,kj->ki',Sq.astype(np.int32),wfn.get_gvec_nk(iqbar)) - G_Sq[np.newaxis,:],dtype=xp.int32)
        V_qfullG.fill(0.0+0.0j)
        V_qfullG[:vcoul_psiG_comps.shape[0]] = V_qG[iqbar,:vcoul_psiG_comps.shape[0]]


        # get correct G components
        #ng_q = int(wfn.ngk[sym.irk_to_k_map[iq]])
        zeta_qG_mu.fill(0.0+0.0j)
        #G_q_comps_cpu = sym.get_gvecs_kfull(wfn,iq)  # Convert to integers. DO -1 ?? TODO
        #G_q_comps = xp.asarray(G_q_comps_cpu).astype(xp.int32)  # Convert to integers. DO -1 ?? TODO
        for mu in range(zeta_q.shape[0]):
            zeta_qG_mu[mu,:vcoul_psiG_comps.shape[0]] = zeta_q[mu,vcoul_psiG_comps[:,0],vcoul_psiG_comps[:,1],vcoul_psiG_comps[:,2]]

        ###############################

        # Store zeta_qG_mu in transposed form initially (ngk, n_rmu)
        # TODO: ascontiguousarray here?
        temp = xp.multiply(V_qfullG[:,None], zeta_qG_mu.T)  # (ngk, n_rmu)
        V_qmunu[*qvec_nonneg] = xp.matmul(xp.conj(zeta_qG_mu), temp)  # (n_rmu, n_rmu)
        #minqvec = [-q for q in qvec] # THIS MAY NOT BE TRUE FOR NON-REAL V(q,G)
        #V_qmunu[*minqvec] = xp.conj(V_qmunu[*qvec]).T # zeta(-q) = zeta*(q)

        print(f"qpoint {iq} done")

    # shape of these is (nkpts, nbands, nspinor, nrmu)
    psi_l_rmu_out = psi_l_rtot_out[:,:,:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]
    psi_r_rmu_out = psi_r_rtot_out[:,:,:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]

    return V_qmunu, xp.conj(psi_l_rmu_out), psi_r_rmu_out


# G_(kab)(mu,nu,t=0) = \sum_mn psi^*_mk(r_mu) * psi_nk(r_nu) (n restricted to range of psi_rmu)
# k goes over kfull
def get_Gk_mu_nu(wfn, psi_l_rmu, psi_r_rmu, n_rmu, xp, Gkij=None):
    # using xp to wrap cupy/numpy, calculate:
    # take the matrix psi with shape (nkpts, nbands, nspinor, nrmu) and do:
    # G_{k,a,b}(mu,nu) = \sum_mnab psi^*_mka(r_mu) * psi_nkb(r_nu) (matmul)
    kgrid = xp.asarray(wfn.kgrid)

    if Gkij is None:
        # saving a dim to include nfreq!
        Gkij = xp.zeros((1,sym.nk_tot, psi_l_rmu.shape[1], psi_r_rmu.shape[1]), dtype=xp.complex128)
        for ik in range(sym.nk_tot):
            xp.fill_diagonal(Gkij[0,ik], 1.0)

    # nspinor*nrmu
    n_spinmu = psi_l_rmu.shape[2]*psi_l_rmu.shape[3]
    # dims: nfreq(=0), nk, n_rmu, n_rmu
    Gk_mu_nu_0 = xp.zeros((1,*wfn.kgrid,n_spinmu,n_spinmu), dtype=xp.complex128)

    for kpt in xp.ndindex(*wfn.kgrid):
        k_idx = kpt[0]*wfn.kgrid[1]*wfn.kgrid[2] + kpt[1]*wfn.kgrid[2] + kpt[2]
        
        psi_l = psi_l_rmu[k_idx,:,:,:].reshape(-1,n_spinmu).T
        psi_r = xp.conj(psi_r_rmu[k_idx,:,:,:]).reshape(-1,n_spinmu)
        Gk_mu_nu_0[0,*kpt] = xp.matmul(xp.matmul(psi_l, Gkij[0,k_idx]), psi_r)

    return Gk_mu_nu_0.reshape(1,*wfn.kgrid,wfn.nspinor,n_rmu,wfn.nspinor,n_rmu)


# get the real-space sigma_\alpha\beta(r,r'(omega))
# options being X, SX, COH
def get_sigma_x_mu_nu(wfn, sym, Gk_mu_nu_0, V_mu_nu, xp):
    # sigma_kbar,ab = \sum_(set of k_i = kbar S_i) \sum_qbar G_(k-qbar,ab)(mu,nu) V_qbar(mu,nu)
    # trying in real space! \sum_R G_R W_R. woohoo
    n_rmu = Gk_mu_nu_0.shape[5]
    n_spinmu = Gk_mu_nu_0.shape[4]*Gk_mu_nu_0.shape[5]
    kgrid = tuple(wfn.kgrid.astype(int))

    # indices here are (nfreq, nkx, nky, nkz, nspin, nrmu, nspin, nrmu)
    # Reorder axes to have kgrid last (batch fft mem locality)
    G_k = xp.ascontiguousarray(
        Gk_mu_nu_0.reshape(1,*kgrid,-1).transpose(0,4,1,2,3)  
        ).reshape(-1, *kgrid) 
    # shape (nfreq*nspin*nrmu*nspin*nrmu, *kgrid)

    # V is umklapped to have kpts in FFT ordering [0,...nk/2,-nk/2+1,...].
    # G doesn't need to be because G_(k+G)(r,r') = G_k(r,r') (bloch fn).
    Gfftplan = cupyx.scipy.fftpack.get_fft_plan(G_k, axes=(1,2,3)) 
    G_R = cupyx.scipy.fftpack.ifftn(G_k, axes=(1,2,3), plan=Gfftplan, overwrite_x=True)
    G_R = G_R.reshape(1,wfn.nspinor,n_rmu,wfn.nspinor,n_rmu,*kgrid)

    V_q = xp.ascontiguousarray(
        V_mu_nu.reshape(-1,n_rmu,n_rmu).transpose(1,2,0)
        ).reshape(-1,*kgrid) 
    # shape (nrmu*nrmu, *kgrid)

    Vfftplan = cupyx.scipy.fftpack.get_fft_plan(V_q, axes=(1,2,3))
    V_R = cupyx.scipy.fftpack.ifftn(V_q, axes=(1,2,3), plan=Vfftplan, overwrite_x=True)
    V_R = V_R.reshape(n_rmu,n_rmu,*kgrid)

    print("G_R and V_R obtained")

    sigma_R = xp.zeros_like(G_R) # shape (nfreq, a, rmu, b, rmu, x,y,z) (a,b spinors)
    sigma_R += G_R * V_R[xp.newaxis,xp.newaxis,:,xp.newaxis,:,:,:,:] 
    sigma_R = sigma_R.reshape(-1,*kgrid) 
    sigma_fftplan = cupyx.scipy.fftpack.get_fft_plan(sigma_R, axes=(1,2,3))
    sigma_k = cupyx.scipy.fftpack.fftn(sigma_R, axes=(1,2,3), plan=sigma_fftplan, overwrite_x=True) # just changed from fftn
    sigma_k = xp.ascontiguousarray(
        sigma_k.reshape(1,n_spinmu*n_spinmu,*kgrid).transpose(0,2,3,4,1)
        ).reshape(1,*kgrid,wfn.nspinor,n_rmu,wfn.nspinor,n_rmu)

    sigma_k *= -sym.nk_tot
    return sigma_k.reshape(1,sym.nk_tot,wfn.nspinor,n_rmu,wfn.nspinor,n_rmu)


def get_sigma_x_kij(psi_l_rmu, psi_r_rmu, sigma_kbar, xp):
    """
    Get sigma matrix elements in band basis by:
    sigma_mnkbar = \sum_rmu,rnu,s,s' exp(ik(r_nu-r_mu)) u_mk^*(r_mu,s) sigma_kbar,ss'(r_mu,r_nu) u_nk(r_nu,s')
    """    
    kgrid = xp.asarray(wfn.kgrid)
    sigma_kij = xp.zeros((sigma_kbar.shape[1], psi_l_rmu.shape[1], psi_r_rmu.shape[1]), 
                        dtype=xp.complex128)
    
    n_spinmu = psi_l_rmu.shape[2]*psi_l_rmu.shape[3]
    #ikrmu = xp.ones(centroids_frac.shape[0], dtype=xp.complex128)

    for kpt in xp.ndindex(*wfn.kgrid):
        k_idx = kpt[0]*wfn.kgrid[1]*wfn.kgrid[2] + kpt[1]*wfn.kgrid[2] + kpt[2]

        sigma_ktmp = sigma_kbar[0,k_idx,:,:,:,:].reshape(n_spinmu,n_spinmu)
    
        # TODO: should probably rearrange psi_l to be contiguous in memory
        psi_l = xp.conj(psi_l_rmu[k_idx,:,:,:]).reshape(-1,n_spinmu)
        psi_r = psi_r_rmu[k_idx,:,:,:].reshape(-1,n_spinmu).T

        sigma_kij[k_idx,:,:] = xp.matmul(xp.matmul(psi_l, sigma_ktmp), psi_r)

    return sigma_kij

def write_sigma_to_file(sigma_kij, filename="eqp0.dat"):
    print(f"sigma_kij dtype before writing: {sigma_kij.dtype}")
    nk, nbands, _ = sigma_kij.shape
    
    with open(filename, 'w') as f:
        for k in range(nk):
            f.write(f"\nk-point {k}:\n")
            f.write("-" * 40 + "\n")
            for n in range(nbands):
                real = float(sigma_kij[k,n,n].real)  # Explicit conversion to float
                imag = float(sigma_kij[k,n,n].imag)
                f.write(f"n={n:<3} {real:>15.6f} + {imag:>15.6f}i\n")

def write_arrays_to_h5(V_qmunu, Gkval_mu_nu, psi_l_rmu_out, psi_r_rmu_out, sigma_x_kbar_munu, sigma_x_kbar_ij, filename="debug_arrays.h5"):
    """Write arrays to HDF5 file for debugging."""
    import h5py
    
    # Convert CuPy arrays to NumPy if needed
    if cp.cuda.is_available():
        V_qmunu = cp.asnumpy(V_qmunu)
        Gkval_mu_nu = cp.asnumpy(Gkval_mu_nu)
        psi_l_rmu_out = cp.asnumpy(psi_l_rmu_out)
        psi_r_rmu_out = cp.asnumpy(psi_r_rmu_out)
        sigma_x_kbar_munu = cp.asnumpy(sigma_x_kbar_munu)
        sigma_x_kbar_ij = cp.asnumpy(sigma_x_kbar_ij)
    
    with h5py.File(filename, 'w') as f:
        # Create groups for organization
        coulomb = f.create_group('coulomb')
        green = f.create_group('green')
        wavefunc = f.create_group('wavefunc')
        sigma = f.create_group('sigma')
        
        # Write arrays with compression
        coulomb.create_dataset('V_qmunu', data=V_qmunu, compression='gzip')
        green.create_dataset('Gkval_mu_nu', data=Gkval_mu_nu, compression='gzip')
        wavefunc.create_dataset('psi_l_rmu_out', data=psi_l_rmu_out, compression='gzip')
        wavefunc.create_dataset('psi_r_rmu_out', data=psi_r_rmu_out, compression='gzip')
        sigma.create_dataset('sigma_x_kbar_munu', data=sigma_x_kbar_munu, compression='gzip')
        sigma.create_dataset('sigma_x_kbar_ij', data=sigma_x_kbar_ij, compression='gzip')
        
        # Add some metadata
        f.attrs['creation_date'] = str(np.datetime64('now'))
        f.attrs['nk_tot'] = sym.nk_tot
        f.attrs['nk_red'] = sym.nk_red
        f.attrs['n_rmu'] = psi_l_rmu_out.shape[3]  # Number of centroids

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
    # Check GPU availability
    if cp.cuda.is_available():
        print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
        mem_info = cp.cuda.runtime.memGetInfo()
        print(f"Memory Usage: {(mem_info[1] - mem_info[0])/1024**2:.1f}MB / {mem_info[1]/1024**2:.1f}MB")
        xp = cp
    else:
        print("Using CPU (NumPy)")
        xp = np

    nval = 5
    ncond = 5
    nband = 30

    sys_dim = 2 # 3 for 3D, 2 for 2D

    ryd2ev = 13.6056980659

    wfn = WFNReader("WFN.h5")
    wfnq = WFNReader("WFNq.h5")
    eps0 = EPSReader("eps0mat.h5")
    eps = EPSReader("epsmat.h5")
    sym = symmetry_maps.SymMaps(wfn)
    q0 = wfnq.kpoints[0] - wfn.kpoints[0]
    if np.linalg.norm(q0) > 1e-6:
        print(f'Using q0 = ({q0[0]:.5f}, {q0[1]:.5f}, {q0[2]:.5f})')

    nvrange, ncrange, nsigmarange, n_fullrange, n_valrange = get_bandranges(nval, ncond, nband, wfn.nelec)

    # Load centroids
    centroids_frac = np.loadtxt('centroids_frac.txt')
    n_rmu = int(centroids_frac.shape[0])

    if cp.cuda.is_available():
        centroids_frac = cp.asarray(centroids_frac, dtype=cp.float32)
        fft_grid = cp.asarray(wfn.fft_grid, dtype=cp.int32)
    centroid_indices = xp.round(centroids_frac * fft_grid).astype(int)
    # Replace any index equal to the grid size with 0 (periodic boundary)
    for i in range(3):
        centroid_indices[centroid_indices[:, i] == wfn.fft_grid[i], i] = 0


    ####################################
    # 1.) get (truncated in 2D) coulomb potential v_q(G) and W_q=0(G=G'=0) element
    ####################################
    V_qG, wcoul0 = get_V_qG(wfn, sym, q0, xp, eps0.epshead, sys_dim)


    ####################################
    # 2.) get interpolative separable density fitting basis functions zeta_q,mu(r)
    # 3.) only store V_q,mu,nu in memory so no need to store all zeta_q,mu(r)
    ####################################
    V_qmunu, psi_l_rmu_out, psi_r_rmu_out = get_zeta_q_and_v_q_mu_nu(wfn, wfnq, sym, centroid_indices, n_valrange, nsigmarange, V_qG, xp)


    #################################### 
    # 4.) get G_k(r_mu,r_nu) for valence bands
    ####################################
    Gkval_mu_nu = get_Gk_mu_nu(wfn, psi_l_rmu_out, psi_l_rmu_out, n_rmu, xp)


    ####################################
    # 5.) get sigma_mnk from V_q,mu,nu and G_k(r_mu,r_nu)
    ####################################
    sigma_x_kbar_munu = get_sigma_x_mu_nu(wfn, sym, Gkval_mu_nu, V_qmunu, xp)
    sigma_x_kbar_ij = get_sigma_x_kij(psi_r_rmu_out, psi_r_rmu_out, sigma_x_kbar_munu, xp)


    write_sigma_to_file(ryd2ev*sigma_x_kbar_ij, "eqp0_noqsym.dat")
    # Call this function after your calculations
    write_arrays_to_h5(V_qmunu, Gkval_mu_nu, psi_l_rmu_out, psi_r_rmu_out, sigma_x_kbar_munu, sigma_x_kbar_ij)
