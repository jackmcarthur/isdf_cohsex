import numpy as np
import cupy as cp
from wfnreader import WFNReader
import fftx
import symmetry_maps
#import matplotlib.pyplot as plt
from interp_vectors_allq import get_interp_vectors_allq#, fft_bandrange_psimu

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


def get_V_qG(wfn, q0, xp, sys_dim):
    # first: V(q,G,G') = 4\pi/|q+G|^2 \delta_{G,G'} * trunc part in 2D, (1-exp(-zc*kxy)*cos(kz*zc))
    #print("vqg start")
    bvec = xp.asarray(wfn.blat * wfn.bvec, dtype=xp.float64)
    q0xp = xp.asarray(q0, dtype=xp.float64)
    qvec = xp.array([xp.float64(0.),xp.float64(0.),xp.float64(0.)])
    #print("vqg qvec done")
    G_q_crys = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    G_cart = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    #print("vqg G_q_crys done")
    V_qG = xp.zeros((int(wfn.nkpts), int(wfn.ngkmax)), dtype=xp.float64)
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
            #print("vqg loop check a")
            #print(f'Getting V_coul for q-vector {iq+1}: ({wfn.kpoints[iq,0]:.4f}, {wfn.kpoints[iq,1]:.4f}, {wfn.kpoints[iq,2]:.4f})',flush=True)
            #print("vqg loop check b")
            Gmax_q = ngks[iq]

            G_q_crys.fill(0)
            G_cart.fill(0)
            # this saves memory in the case of many kpts but requires a lot of HtoD transfers. revisit.
            G_q_crys[:Gmax_q] = xp.asarray(wfn.get_gvec_nk(iq).astype(np.float64)) # stored as int32, trying to convert efficiently

            #G_cart = xp.einsum('ik,jk->ji', bvec.T, G_q_crys + qvec) # note bvec includes blat
            G_cart[:Gmax_q] = xp.matmul(G_q_crys[:Gmax_q] + qvec, bvec) # @ is super slow, probably using numpy
            #print("done with gcart")
            #V_qG[iq,:Gmax_q] = 4*xp.pi/ xp.einsum('ij,ij->i',G_cart,G_cart)[:Gmax_q]
            V_qG[iq,:Gmax_q] = xp.divide(4*xp.pi, xp.sum(G_cart*G_cart, axis=1)[:Gmax_q])
            #print("done with vqg no trunc")
            kxy = xp.linalg.norm(G_cart[:Gmax_q,:2], axis=1)
            kz = G_cart[:Gmax_q,2]
            zc = xp.pi/bvec[2,2] # note that the crystal z axis must align with the cartesian z axis
            # NOT SURE WHY THERES AN EXTRA 2. 8PI NOT 4PI? I\neq J probably? but i compared to an epsmat.h5 file
            V_qG[iq,:Gmax_q] *= 2 * (1-xp.exp(-zc*kxy)*xp.cos(kz*zc))

        # mini-BZ monte carlo integration for V_q=0,G=0
        randvals = xp.random.rand(1000000,3) - 0.5
        randlims = 1.0 / xp.asarray(wfn.kgrid)
        randqs = randvals * randlims # set of non-grid qpts closer to q=0 than any other qpt
        randqcart = xp.einsum('ik,jk->ji', bvec.T, randqs)
        rand_vq = 4*xp.pi / xp.einsum('ij,ij->i',randqcart,randqcart)
        rand_vq *= 2 * (1 - xp.exp(-xp.pi/bvec[2,2] * xp.linalg.norm(randqcart[:,:2], axis=1)) * xp.cos(randqcart[:,2] * xp.pi/bvec[2,2]))

        #print(f"V_q=0,G=0 from q0: {V_qG[0,0]:.4f}")
        V_qG[0,0] = xp.mean(rand_vq)
        #print(f"V_q=0,G=0 from miniBZ monte carlo: {V_qG[0,0]:.4f}")

    return V_qG

def unfold_V_qG(V_qbarG, sym):
    V_qfullG = xp.zeros((sym.nk_tot, V_qbarG.shape[1]), dtype=xp.float64)
    for iqfull in range(sym.nk_tot):
        qbar = sym.irk_to_k_map[iqfull]

        qbar_gvecs = sym.get_gvec_nk(qbar)
        qfull_gvecs = sym.get_gvec_nk(iqfull)

        # both q_gvecs are shape (ngk,3), lists of 3d int coords like [0,0,0],[1,-1,2], etc.
        # here, get the indices of 


        V_qfullG[iqfull] = V_qbarG
        pass

def unfold_zeta_q(zeta_q, sym_mat_q):
    # unfold zeta_q(G) to zeta_qfull(G) where qfull = qbar S_i and can be outside the 1bz
    pass

def get_V_mu_nu(wfn, zeta_q, V_qG, xp):
    # here: calculate <zeta_q_mu|V(q,G,G')|zeta_q_nu>
    # we attempt to do this as \sum_G zeta*_qmu(-G) * v(q,G) * zeta_qnu(-G) (negative bc of how the FT is defined)?
    # first: FFT zeta_q(r) to G space (shape nk, nrmu, *nfft)
    print(f"zeta norm pre fft {np.linalg.norm(zeta_q[0,0])}")
    for iq in range(zeta_q.shape[0]):
        for mu in range(zeta_q.shape[1]):
            zeta_q[iq,mu] = fftx.fft.fftn(zeta_q[iq,mu])
    #print(zeta_q.shape[1] * zeta_q.shape[2] * zeta_q.shape[3])
    fft_vol = xp.asarray([zeta_q.shape[2] * zeta_q.shape[3] * zeta_q.shape[4]], dtype=xp.float32)
    #print(f"FFT volume: {fft_vol}")
    zeta_q *= xp.sqrt(1./fft_vol)  # normalize FFT
    print(f"zeta norm post fft {np.linalg.norm(zeta_q[0,0])}")
    # get correct G components
    zeta_qG_mu = xp.zeros((zeta_q.shape[0], zeta_q.shape[1], int(wfn.ngkmax)), dtype=xp.complex128)
    #zeta_qG_nu = xp.zeros((zeta_q.shape[0], zeta_q.shape[1], int(wfn.ngkmax)), dtype=xp.complex128)
    #G_q_comps = xp.zeros((int(wfn.ngkmax), 3), dtype=int)
    for iq in range(zeta_q.shape[0]):
            G_q_comps = xp.asarray(-1*wfn.get_gvec_nk(iq))
            # there should NOT need to be a transpose here.
            for mu in range(zeta_q.shape[1]):
                zeta_qG_mu[iq,mu,:G_q_comps.shape[0]] = zeta_q[iq,mu,G_q_comps[:,0],G_q_comps[:,1],G_q_comps[:,2]]
            #for nu in range(zeta_q.shape[1]):
            #    zeta_qG_nu[iq,nu,:G_q_comps.shape[0]] = zeta_q[iq,nu,G_q_comps[:,0],G_q_comps[:,1],G_q_comps[:,2]]

    print(zeta_qG_mu.shape)
    print(V_qG.shape)

    # zeta_qG_flat is shape (2, ngk, n_rmu)
    # V_qG is shape (nk, ngk)
    # do V_\mu\nu = \int dG zeta_qG_flat[iq,mu,G] * V_qG[iq,G] * zeta_qG_flat.T[iq,nu,G], vectorized
    Vq_mu_nu = xp.einsum('qig,qg,qjg->qij', 
                        xp.conj(zeta_qG_mu),  # (nq,n_rmu,ngk)
                        V_qG,  # (nq,ngk)
                        zeta_qG_mu)           # (nq,n_rmu,ngk)
    print('done with Vq_mu_nu')
    return Vq_mu_nu

# G_(kab)(mu,nu,t=0) = \sum_n psi^*_nk(r_mu) * psi_nk(r_nu) (n restricted to range of psi_rmu)
# k goes over kfull
def get_Gk_mu_nu_0(psi_rmu, n_rmu, xp):
    # using xp to wrap cupy/numpy, calculate:
    # take the matrix psi with shape (nkpts, nbands, nspinor, nrmu) and do:
    # G_{k,a,b}(mu,nu) = \sum_mnab psi^*_mka(r_mu) * psi_nkb(r_nu) (matmul)

    n_spinmu = psi_rmu.shape[2]*psi_rmu.shape[3]
    # dims: nfreq(=0), nk, n_rmu, n_rmu
    Gk_mu_nu_0 = xp.zeros((1,psi_rmu.shape[0],n_spinmu,n_spinmu), dtype=xp.complex128)

    for nk in xp.ndindex(psi_rmu.shape[0]):
        psi = psi_rmu[nk,:,:,:].reshape(-1,n_spinmu)
        Gk_mu_nu_0[0,nk,:,:] = xp.matmul(xp.conj(psi).T, psi)

    return Gk_mu_nu_0.reshape(1,sym.nk_tot,2,n_rmu,2,n_rmu)


# get the real-space sigma_\alpha\beta(r,r'(omega)) for some specific sigma contribution
# options being X, SX, COH
def get_sigma_x_mu_nu(wfn, sym, Gk_mu_nu_0, V_mu_nu, xp):
    # sigma_kbar,ab = \sum_(set of k_i = kbar S_i) \sum_qbar G_(k-qbar,ab)(mu,nu) V_qbar(mu,nu)

    sigma_kbar = xp.zeros_like(Gk_mu_nu_0) # shape (nfreq, nkbar, a, rmu, b, rmu) (a,b spinors)

    for ikfull in range(sym.nk_tot):
        for iqbar in range(wfn.nkpts):
            kfullminusqbar = sym.kq_map[ikfull, iqbar] # CHANGED KQ MAP BELOW, OUT OF DATE
            G_kminq = Gk_mu_nu_0[0,kfullminusqbar,:,:,:,:]
            sigma_kbar[0,iqbar,:,:,:,:] += xp.einsum('anbl,nl->anbl', G_kminq, V_mu_nu[iqbar])

    sigma_kbar *= -1./sym.nk_tot
    return sigma_kbar

def get_sigma_x_kij(psi_rmu, sigma_kbar, xp):
    # sigma_kbar,ij = \sum_a,mu,b,nu psi_rmu^*[kbar,i,a,mu] * sigma_kbar[0,kbar,a,mu,b,nu] * psi_rmu[kbar,j,b,nu]
    # output: shape (nkpts, nbands, nbands)
    sigma_kij = xp.zeros((sigma_kbar.shape[1], psi_rmu.shape[1], psi_rmu.shape[1]), dtype=xp.complex128)

    n_spinmu = psi_rmu.shape[2]*psi_rmu.shape[3]
    for ikbar in range(wfn.nkpts):
        sigma_ktmp = sigma_kbar[0,ikbar,:,:,:,:].reshape(n_spinmu,n_spinmu)
        psi_mu = psi_rmu[ikbar,:,:,:].reshape(-1,n_spinmu)
        sigma_kij[ikbar,:,:] = xp.matmul(xp.matmul(xp.conj(psi_mu), sigma_ktmp), psi_mu.T)

    return sigma_kij

def sigma_kij_from_gmunu(wfn, sym, Gk_mu_nu_0, V_mu_nu, psi_rmu, psi_sigma_rmu, xp):
    # it is slightly a pain in the ass to get sigma_kbar with symmetries because
    # \sigma_mnkbar = \sum_S \sum_mu,nu \psi*_kbarS,m(r_mu) \psi_kbarS,n(r_nu) \sum_qbar G_(kbarS-qbar)(r_mu,r_nu) V_(qbar,mu,nu)
    # where there is a sum over the little group outside the whole thing, and there is no quantity \sigma_kbar,mu,nu, only \sigma_kbar,mu,nu(S)
    n_spinmu = psi_rmu.shape[2]*psi_rmu.shape[3]
    sigma_ijkbar = xp.zeros((wfn.nkpts, psi_sigma_rmu.shape[1], psi_sigma_rmu.shape[1]), dtype=xp.complex128)
    sigma_munuS_tmp = xp.zeros((n_spinmu, n_spinmu), dtype=xp.complex128)

    for ikbar in range(wfn.nkpts): # kbar in sigma_kbar
        #for kfull in sym.irk_to_k_map[ikbar]: # kfull = kbarS in wfns and G
        for iqbar in range(sym.nk_tot): # qbar in G and W convolution
            sigma_munuS_tmp.fill(0)
            for iqfull in np.where(sym.irk_to_k_map == iqbar)[0]:
                # Debug prints
                #print(f"iqfull: {iqfull}")
                #print(f"kfull_symmap[iqfull]: {sym.kfull_symmap[iqfull]}")
                matches = np.where(sym.kfull_symmap[iqfull] == sym.irk_to_k_map[iqbar][0])[0] # second is whatever iqbar*identity is in the full klist
                #print(f"matches: {matches}")
                
                if len(matches) == 0:
                    raise ValueError(f"No symmetry operation found mapping k-point {iqfull} to {iqbar}")
                    
                isymkfull = sym.irk_sym_map[iqfull]  #matches[0]
                kfull = sym.kfull_symmap[ikbar, isymkfull]
                kfullminusqbar = sym.kqfull_map[kfull, iqfull]
                #print(f"kfullminusqbar: {kfullminusqbar}")
                kminusqfull = sym.kq_map[ikbar, iqfull]
                qbar = sym.irk_to_k_map[kminusqfull]
                G_kminq = Gk_mu_nu_0[0,kminusqfull,:,:,:,:]
                sigma_munuS_tmp += xp.einsum('anbl,nl->anbl', G_kminq, V_mu_nu[qbar]).reshape(n_spinmu,n_spinmu)
            
            psi_mu = psi_sigma_rmu[kfull,:,:,:].reshape(-1,n_spinmu)
            sigma_ijkbar[ikbar,:,:] += xp.matmul(xp.matmul(xp.conj(psi_mu), sigma_munuS_tmp), psi_mu.T)
        # for iqfull in range(sym.nk_tot):
        #     kbarminusqfull = sym.kqfull_map[ikbar, iqfull]
        #     iqbar = sym.kpoint_map[kbarminusqfull]
        #     G_kminq = Gk_mu_nu_0[0,kbarminusqfull,:,:,:,:]
        #     sigma_munuS_tmp += xp.einsum('anbl,nl->anbl', G_kminq, V_mu_nu[iqbar]).reshape(n_spinmu,n_spinmu)
        # psi_mu = psi_sigma_rmu[ikbar,:,:,:].reshape(-1,n_spinmu)
        # sigma_ijkbar[ikbar,:,:] += xp.matmul(xp.matmul(xp.conj(psi_mu), sigma_munuS_tmp), psi_mu.T)

    sigma_ijkbar *= -1./sym.nk_tot
    return sigma_ijkbar

def print_sigma_matrix(sigma_kij, k_idx, n_elements=5):
    """Print sigma matrix elements in a nicely formatted table.
    
    Args:
        sigma_kij: Complex array of shape (nk, nbands, nbands)
        k_idx: k-point index to print
        n_elements: Number of diagonal elements to print (default 5)
    """
    print(f"\nk-point {k_idx}:")
    print("-" * 40)
    for n in range(n_elements):
        real = sigma_kij[k_idx,n,n].real
        imag = sigma_kij[k_idx,n,n].imag
        print(f"n={n:<2} {real:>10.5f} + {imag:>10.5f}i")

def write_sigma_to_file(sigma_kij, filename="eqp0.dat"):
    """Write sigma matrix diagonal elements to file for all k-points and bands.
    
    Args:
        sigma_kij: Complex array of shape (nk, nbands, nbands)
        filename: Output filename (default "eqp0.dat")
    """
    nk, nbands, _ = sigma_kij.shape
    
    with open(filename, 'w') as f:
        for k in range(nk):
            f.write(f"\nk-point {k}:\n")
            f.write("-" * 40 + "\n")
            for n in range(nbands):
                real = sigma_kij[k,n,n].real
                imag = sigma_kij[k,n,n].imag
                f.write(f"n={n:<3} {real:>10.5f} + {imag:>10.5f}i\n")



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

    # loop over q-vecs to get contributions to sigma at every reduced k-point
    # for iq in range(wfn.nkpts):
        
    # Get interpolation vectors for \sigma_X
    # shape of zeta:nspinor (full grid), nfftx, nffty, nfftz, 2*n_rmu
    #iq = 0
    #zeta_q, psi_val_mu, psi_sigma_mu, psi_val_rtot, psi_sigma_rtot = get_interp_vectors_q(wfn, wfnq, sym, iq, centroid_indices, nsigmarange, n_valrange, xp)

    zeta_q, psi_l_rmu_out, psi_r_rmu_out, psi_l_rtot_out, psi_r_rtot_out = get_interp_vectors_allq(wfn, wfnq, sym, centroid_indices, nsigmarange, n_valrange, xp)

    V_qG = get_V_qG(wfn, q0, xp, sys_dim)
    V_mu_nu = get_V_mu_nu(wfn, zeta_q, V_qG, xp)

    bandrange_tot = (0, wfn.nelec)
    psi_rmu = xp.zeros((sym.nk_tot,wfn.nelec, 2, n_rmu), dtype=xp.complex128)
    psi_sigma_rmu = xp.zeros((sym.nk_tot,nsigmarange[1]-nsigmarange[0], 2, n_rmu), dtype=xp.complex128)
    #Gmnk = xp.zeros((sym.nk_tot, wfn.nbands, wfn.nbands), dtype=xp.complex128)
    #fft_bandrange_psimu(wfn, sym, bandrange_tot, centroid_indices, psi_rmu, xp=cp)
    #fft_bandrange_psimu(wfn, sym, nsigmarange, centroid_indices, psi_sigma_rmu, xp=cp)
    psi_rmu = psi_r_rmu_out
    psi_sigma_rmu = psi_l_rmu_out

    Gk_mu_nu_0 = get_Gk_mu_nu_0(psi_rmu, n_rmu, xp)
    #sigma_munu = get_sigma_x_mu_nu(wfn, sym, Gk_mu_nu_0, V_mu_nu, xp)
    #sigma_x_kij = get_sigma_x_kij(psi_sigma_rmu, sigma_munu, xp)
    sigma_x_kij = sigma_kij_from_gmunu(wfn, sym, Gk_mu_nu_0, V_mu_nu, psi_rmu, psi_sigma_rmu, xp)
    print(sigma_x_kij[0])
    
    # get V_\mu,\nu
    # pass zeta_iq 

    # Call it for first k-point
    print_sigma_matrix(sigma_x_kij, 0)

    # Write all k-points and bands to file
    write_sigma_to_file(sigma_x_kij)

    print(1+1)




