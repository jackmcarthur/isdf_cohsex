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


def get_static_chi_q(wfn, sym, G_R, xp):
    # chi(r,r',t=0) = G_R(r,r',t=0)G_-R(r',r,t=0)
    n_rmu = G_R.shape[2]
    n_spinor = G_R.shape[1]
    kgrid = tuple(G_R.shape[5:])
    nk = np.prod(kgrid)

    chi_R = xp.zeros_like(G_R)

    G_R = G_R.reshape(1,n_spinor*n_rmu, n_spinor*n_rmu, *kgrid)
    G_negR = xp.ascontiguousarray(G_R.transpose(0,2,1,3,4,5))
    # handle kx ky kz
    for ik in range(3,6):
        G_negR = xp.flip(G_negR, axis=ik)
        G_negR = xp.roll(G_negR,1, axis=ik)
    # handle frequency
    G_negR = xp.flip(G_negR, axis=0)
    G_negR = xp.roll(G_negR,1, axis=0)

    chi_R = G_R * G_negR
    chi_q = cupyx.scipy.fftpack.fftn(chi_R, axes=(3,4,5), overwrite_x=True, norm='ortho')
    chi_q = chi_q.reshape(1,n_spinor,n_rmu,n_spinor,n_rmu,*kgrid)

    return chi_q


def get_static_w_q(chi_R, V_qmunu, sym, xp):
    # w_q(omega) = (1-v_q chi_q)^{-1} v_q

    w_q = xp.zeros_like(chi_R) # has spinor components
