import numpy as np
import cupy as cp
from wfnreader import WFNReader
from tagged_arrays import LabeledArray
# The routines here construct chi^0 and the screened interaction W using the
# CTSP approach in the static limit.  Once the frequency grids are restored, the
# same machinery will let us tackle full dynamical GW.
try:
    cp.cuda.runtime.getDeviceCount()
    xp = cp
except Exception:
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
        # Mask for energies within the valence window (windows are inclusive)
        val_mask = (psi_v.enk.data[ik] >= win.val_window.start_energy) & (psi_v.enk.data[ik] <= win.val_window.end_energy)
        
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
        cond_mask = (psi_c.enk.data[ik] >= win.cond_window.start_energy) & (psi_c.enk.data[ik] <= win.cond_window.end_energy)
        
        psi_c_selected = psi_c.psi.data[ik, cond_mask]
        exp_c_selected = exp_tauE_c[:,ik, cond_mask]
        
        psi_c_conj = xp.conj(psi_c_selected)
        M = xp.einsum('tb,bi->tib', exp_c_selected, psi_c_selected)
        Gc_lm.data[ik] = xp.matmul(M, psi_c_conj)

    Gv_lm.unjoin('nkx', 'nky', 'nkz')
    Gv_lm = Gv_lm.kgrid_to_last()
    Gv_lm.ifft_kgrid() # G_k -> G_R

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

    # note it would be more efficient to only fft chi0 in get_chi0
    chi_lm_Yt.fft_kgrid() # chi_R -> chi_q
    chi_out = chi_lm_Yt.transpose('ntau', 'nkx', 'nky', 'nkz', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2')
    oneoverkgrid = xp.complex128(np.power(np.complex128(wfn.kgrid[0]*wfn.kgrid[1]*wfn.kgrid[2]),-1))
    xp.multiply(chi_out.data, oneoverkgrid, out=chi_out.data)
    print('one chi_lm element ', chi_out.data[0,0,0,0,0,0,0,0].get())
    return chi_out.data


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
        quad_weights = xp.asarray(-2.*win.z_lm*win.w_i*np.exp(-(win.z_lm*(win.cond_window.start_energy - win.val_window.end_energy)-1.)*win.tau_i),dtype=xp.complex128)
        # note that doing += doesn't work because it's not a cupy fn 
        xp.add(chi0.data[0,:,:,:,:,:,:,:], xp.einsum('t,ti...->i...', quad_weights, chi_lm), out=chi0.data[0,:,:,:,:,:,:,:])

    chi = chi0.transpose('nkx', 'nky', 'nkz', 'ntau', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2')
    return chi

def get_static_w_q(chi_q, V_q, wfn, sym, xp, n_mult=10, block_f=1):
    # w_q(omega) = (1-v_q @ chi_q)^{-1} @ v_q
    # This implementation performs the CTSP matrix inversion in the static limit.
    # Once the frequency mesh is restored this routine will compute W(omega) on
    # the full imaginary-time grid.
    # if A = v_q @ chi_q, then (1-A)^{-1} = 1 + A + A^2 + A^3 + ... (iterative matrix inversion faster + more stable than direct)
    # A^N is done with blocked GEMMs along the frequency axis; since we currently do COHSEX we set block_q=1
    nspinor_w = chi_q.shape('nspinor1')
    nrmu = chi_q.shape('nrmu1')
    print('one chi element: ', chi_q.data[0,0,0,0,0,0,0,0].get())

    V_q.join('nkx', 'nky', 'nkz')
    V_q.join('nspinor1', 'nrmu1')
    V_q.join('nspinor2', 'nrmu2')

    chi_q.join('nkx', 'nky', 'nkz')
    chi_q.join('nspinor1', 'nrmu1')
    chi_q.join('nspinor2', 'nrmu2')

    W_q = LabeledArray(shape=(*wfn.kgrid, 1, nspinor_w, nrmu, nspinor_w, nrmu), axes=('nkx', 'nky', 'nkz', 'nfreq','nspinor1', 'nrmu1', 'nspinor2', 'nrmu2'))
    W_q.join('nkx', 'nky', 'nkz')
    W_q.join('nspinor1', 'nrmu1')
    W_q.join('nspinor2', 'nrmu2')
    
    nk_tot, nfreq, N, _ = chi_q.data.shape

    # pick a block‐size along the frequency axis
    if block_f is None:
        # e.g. cap at 128 MB of scratch:
        max_bytes = 128 * 1024**2
        per_mat   = 16 * N * N       # bytes per (N×N) complex128
        block_f   = max(1, int(max_bytes // per_mat))
    block_f = min(block_f, nfreq)

    # allocate scratch buffers once
    A   = xp.empty((block_f, N, N), dtype=xp.complex128)
    Wb  = xp.empty((block_f, N, N), dtype=xp.complex128)
    P   = xp.empty((block_f, N, N), dtype=xp.complex128)
    I   = xp.eye(N, dtype=xp.complex128)[None, :, :]

    # loop over q‐points
    for iq in range(nk_tot):
        Vf = V_q.data[0,iq]  # shape = (N, N)
        ch = chi_q.data[iq]  # shape = (nfreq, N, N)
        Wf = W_q.data[iq]    # shape = (nfreq, N, N)

        # chunk over freq‐axis
        for f0 in range(0, nfreq, block_f):
            f1 = min(f0+block_f, nfreq)
            B  = f1 - f0

            cb = ch[f0:f1]      # (B, N, N)
            wb = Wb[:B]         # view into scratch
            a  = A[:B]

            # 1) A := Vb @ cb
            xp.matmul(Vf, cb, out=a)

            # 2) Wb := I + A
            wb[:] = I           # broadcast eye
            wb += a

            # 3) Build powers A^2 … A^(n_mult+1)
            # P = a.copy()        # P == A^1
            # for _ in range(n_mult):
            #     xp.matmul(P, cb, out=P)
            #     wb += P
            cb = a.copy() # chi array now contains vchi
            for _ in range(n_mult-1):
                xp.matmul(cb, a, out=P)
                wb += P
                cb = P.copy()
                print('mtx norm P: ', xp.linalg.norm(P))
            # 4) Multiply by Vb → W = (1 - Vχ)^(-1) V
            xp.matmul(wb, Vf, out=Wf[f0:f1])

            # 5) write‐back
            #Wf[f0:f1] = wb

    W_q.unjoin('nkx', 'nky', 'nkz')
    #W_q.kgrid_to_last()
    #W_q.ifft_kgrid() # W_q -> W_R
    W_q.unjoin('nspinor1', 'nrmu1')
    W_q.unjoin('nspinor2', 'nrmu2')
    # could do W_q -> W_R here but it's already done in the get_sigma function
    W = W_q.transpose('nfreq', 'nkx', 'nky', 'nkz', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2')

    V_q.unjoin('nkx', 'nky', 'nkz')
    V_q.unjoin('nspinor1', 'nrmu1')
    V_q.unjoin('nspinor2', 'nrmu2')

    return W



