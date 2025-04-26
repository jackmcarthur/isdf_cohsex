import numpy as np
import cupy as cp
from wfnreader import WFNReader
from tagged_arrays import LabeledArray
#import matplotlib.pyplot as plt
if cp.cuda.is_available():
    xp = cp
else:
    xp = np

# do chi_lm,0(r,r',Yt) = \sum_ab Gc_lm,R(ra,r'b,Yt)Gv_lm,-R(r'b,ra,-Yt) (a,b=spin indices)
def get_chi_lm_Yt(psi_v, psi_c, win, wfn, xp):
    ntau = win.ntau
    nspinor = psi_v.psi.shape('nspinor')
    nrmu = psi_v.psi.shape('nrmu')
    psi_v.psi.join('nspinor', 'nrmu')
    psi_c.psi.join('nspinor', 'nrmu')
    # getting G(A)_lm and G(B)_lm for each tau
    Gv_lm = LabeledArray(shape=(*wfn.kgrid, ntau, nspinor, nrmu, nspinor, nrmu), axes=('nkx', 'nky', 'nkz', 'ntau', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2'))
    Gc_lm = LabeledArray(shape=(*wfn.kgrid, ntau, nspinor, nrmu, nspinor, nrmu), axes=('nkx', 'nky', 'nkz', 'ntau', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2'))
    chi_lm_Yt = LabeledArray(shape=(ntau, 1, nrmu, 1, nrmu, *wfn.kgrid), axes=('ntau', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2', 'nkx', 'nky', 'nkz'))
    Gv_lm.join('nkx', 'nky', 'nkz')
    Gc_lm.join('nkx', 'nky', 'nkz')
    Gv_lm.join('nspinor1', 'nrmu1')
    Gv_lm.join('nspinor2', 'nrmu2')
    Gc_lm.join('nspinor1', 'nrmu1')
    Gc_lm.join('nspinor2', 'nrmu2')

    # these *should* have shape nk,ntau, nb
    exp_tauE_c = xp.exp(-win.z_lm *xp.asarray(win.tau_i[:,np.newaxis,np.newaxis]) * (psi_c.enk.data[np.newaxis,:,:] - win.cond_window.start_energy))
    exp_tauE_v = xp.exp(-win.z_lm *xp.asarray(win.tau_i[:,np.newaxis,np.newaxis]) * (win.val_window.end_energy - psi_v.enk.data[np.newaxis,:,:]))

    # need to loop over 'nk','nb' in psi_v then psi_c
    # for each ik, for each ib, if win.val_window.start_energy < psi_v.enk.data[ik,ib] < win.val_window.end_energy:
    # add to Gv_lm.data[ik,:,:,:] the outer product of psi_v.wfn.data[ik,ib,:] with xp.conj(psi_v.wfn.data[ik,ib,:]).T into the last two axes, duplicated along the first (tau) axis, multiplied by exp_tauE_v[tau,ik,ib] for each tau

    # Iterate over k-points
    for ik in range(psi_v.psi.shape('nk')):
        # Mask for energies within the valence window
        val_mask = (psi_v.enk.data[ik] > win.val_window.start_energy) & (psi_v.enk.data[ik] <= win.val_window.end_energy)
        
        # Select wavefunctions and energies for the current k-point
        psi_v_selected = psi_v.psi.data[ik,val_mask]
        exp_v_selected = exp_tauE_v[:,ik,val_mask]
        
        # Compute outer products and update Gv_lm
        psi_v_conj = xp.conj(psi_v_selected)
        # two step process for G_k is more mem efficient than a single triple einsum.
        M = xp.einsum('tb,bi->tib', exp_v_selected, psi_v_conj)
        # this does a batched GEMM over t dimension:
        Gv_lm.data[ik] = xp.matmul(M, psi_v_selected)

    # Similar process for conduction bands
    for ik in range(psi_c.psi.shape('nk')):
        cond_mask = (psi_c.enk.data[ik] > win.cond_window.start_energy) & (psi_c.enk.data[ik] <= win.cond_window.end_energy)
        
        psi_c_selected = psi_c.psi.data[ik, cond_mask]
        exp_c_selected = exp_tauE_c[:,ik, cond_mask]
        
        psi_c_conj = xp.conj(psi_c_selected)
        M = xp.einsum('tb,bi->tib', exp_c_selected, psi_c_selected)
        Gc_lm.data[ik] = xp.matmul(M, psi_c_conj)

    Gv_lm.unjoin('nkx', 'nky', 'nkz')
    Gv_lm = Gv_lm.kgrid_to_last()
    Gv_lm.ifft_kgrid()

    Gc_lm.unjoin('nkx', 'nky', 'nkz')
    Gc_lm = Gc_lm.kgrid_to_last()
    Gc_lm.ifft_kgrid()


    psi_v.psi.unjoin('nspinor', 'nrmu')
    psi_c.psi.unjoin('nspinor', 'nrmu')

    # flip Gv_R -> Gv_-R, keeping Gv_R=0 in the 0th index
    for ik in range(3,6):
        Gv_lm.data = xp.flip(Gv_lm.data, axis=ik)
        Gv_lm.data = xp.roll(Gv_lm.data,1, axis=ik)

    Gv_lm.unjoin('nspinor1', 'nrmu1')
    Gv_lm.unjoin('nspinor2', 'nrmu2')
    Gc_lm.unjoin('nspinor1', 'nrmu1')
    Gc_lm.unjoin('nspinor2', 'nrmu2')

    for a in range(psi_v.psi.shape('nspinor')):
        for b in range(psi_v.psi.shape('nspinor')):
            chi_lm_Yt.data[:,0,:,0,:,:,:,:] += xp.multiply(Gv_lm.slice_many({'nspinor1':a,'nspinor2':b}), Gc_lm.slice_many({'nspinor1':b,'nspinor2':a}))

    chi_lm_Yt.fft_kgrid()
    chi_lm_Yt = chi_lm_Yt.transpose('ntau', 'nkx', 'nky', 'nkz', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2')
    # todo: this transpose isn't working. ntau possibly mixed up with nkx,nky,nkz
    return chi_lm_Yt


# sums contributions from all windows
def get_chi0(psi_v, psi_c, windows, wfn, xp):
    nspinor = psi_v.psi.shape('nspinor')
    nrmu = psi_v.psi.shape('nrmu')
    chi0 = LabeledArray(shape=(1, *wfn.kgrid, 1, nrmu, 1, nrmu), axes=('ntau', 'nkx', 'nky', 'nkz', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2'))
    #chi0.join('nkx', 'nky', 'nkz')
    #chi0.join('nspinor1', 'nrmu1')
    #chi0.join('nspinor2', 'nrmu2')

    for win in windows:
        chi_lm = get_chi_lm_Yt(psi_v, psi_c, win, wfn, xp)
        # -2 z_lm w_i exp(-(z_lm (E_c - E_v) - 1) tau_i)
        quad_weights = xp.asarray(-2*win.z_lm*win.w_i*np.exp(-(win.z_lm * (win.cond_window.start_energy - win.val_window.end_energy)-1.)*win.tau_i),dtype=xp.complex128)

        chi0.data[0] += xp.einsum('t,ti...->i...', quad_weights, chi_lm.data)

    return chi0

# def get_static_chi_q(wfn, sym, G_R, xp):
#     # chi(r,r',t=0) = G_R(r,r',t=0)G_-R(r',r,t=0)
#     n_rmu = G_R.shape[2]
#     n_spinor = G_R.shape[1]
#     kgrid = tuple(G_R.shape[5:])
#     nk = np.prod(kgrid)

#     chi_R = xp.zeros_like(G_R)

#     G_R = G_R.reshape(1,n_spinor*n_rmu, n_spinor*n_rmu, *kgrid)
#     G_negR = xp.ascontiguousarray(G_R.transpose(0,2,1,3,4,5))
#     # handle kx ky kz
#     for ik in range(3,6):
#         G_negR = xp.flip(G_negR, axis=ik)
#         G_negR = xp.roll(G_negR,1, axis=ik)
#     # handle frequency
#     G_negR = xp.flip(G_negR, axis=0)
#     G_negR = xp.roll(G_negR,1, axis=0)

#     chi_R = G_R * G_negR
#     chi_q = cupyx.scipy.fftpack.fftn(chi_R, axes=(3,4,5), overwrite_x=True, norm='ortho')
#     chi_q = chi_q.reshape(1,n_spinor,n_rmu,n_spinor,n_rmu,*kgrid)

#     return chi_q


# def get_static_w_q(chi_R, V_qmunu, sym, xp):
#     # w_q(omega) = (1-v_q chi_q)^{-1} v_q

#     w_q = xp.zeros_like(chi_R) # has spinor components




# if __name__ == "__main__":
#     wfn = WFNReader("WFN.h5")
#     sym = symmetry_maps.SymmetryMap(wfn.structure)
#     G_R = wfn.get_G_R()
#     chi_q = get_static_chi_q(wfn, sym, G_R, xp)
#     print(chi_q.shape)

