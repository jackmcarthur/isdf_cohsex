import types
import sys
import os
import numpy as np

# Ensure repository root is on the module search path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Create a minimal CuPy stub so modules depending on CuPy import without error
cupy_stub = types.ModuleType('cupy')
# expose numpy functionality used in get_windows
for name in [
    'exp', 'sqrt', 'linspace', 'asarray', 'sum', 'zeros', 'arange']:
    setattr(cupy_stub, name, getattr(np, name))
# minimal CUDA runtime interface
cupy_stub.cuda = types.SimpleNamespace(
    runtime=types.SimpleNamespace(getDeviceCount=lambda: 0)
)
cupy_stub.complex128 = np.complex128
sys.modules.setdefault('cupy', cupy_stub)

import wfnreader
import get_windows


def test_gauss_laguerre_transform():
    """Validate CTSP transform against analytic result."""
    wfn = wfnreader.WFNReader('WFN.h5')
    windows = get_windows.get_window_info(0.01, wfn)
    win = windows[0]

    # Example energies: top valence and lowest conduction
    E_v = wfn.energies[0, 0, wfn.nelec - 1]
    E_c = wfn.energies[0, 0, wfn.nelec]

    E_gap = win.cond_window.start_energy - win.val_window.end_energy
    z_lm = win.z_lm

    # Quadrature approximation using same weights/exponentials as w_isdf.py
    tau_i = win.tau_i
    w_i = win.w_i
    F_quad = -2 * z_lm * np.sum(w_i * np.exp(-(z_lm * (E_c - E_v + E_gap) - 2) * tau_i))

    # Analytic integral of Eq. (24)
    F_exact = -2 * z_lm / (z_lm * (E_c - E_v + E_gap) - 1)

    assert np.isclose(F_quad, F_exact, rtol=1e-7, atol=1e-9)
