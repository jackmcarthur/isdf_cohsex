import os
import sys
import types
import numpy as np

# Ensure repository root is on the module search path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Provide a minimal CuPy stub so modules depending on CuPy import without error
cupy_stub = types.ModuleType("cupy")
for name in [
    "exp",
    "sqrt",
    "linspace",
    "asarray",
    "sum",
    "zeros",
    "arange",
]:
    setattr(cupy_stub, name, getattr(np, name))
cupy_stub.linalg = types.SimpleNamespace(norm=np.linalg.norm)
cupy_stub.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 0))
cupy_stub.complex128 = np.complex128
cupy_stub.float64 = np.float64
sys.modules.setdefault("cupy", cupy_stub)

import wfnreader
from get_windows import get_window_info


def test_ctsp_transform_matches_exact():
    """Validate the CTSP Gauss-Laguerre transform against its analytic form."""

    wfn = wfnreader.WFNReader("WFN.h5")

    # Obtain the first valence/conduction window pair which sets z_lm and tau grid
    win = get_window_info(0.01, wfn)[0]

    energies = wfn.energies[0, 0]
    E_v = energies[wfn.nelec - 1]
    E_c = energies[wfn.nelec]

    z = win.z_lm
    tau = win.tau_i
    w = win.w_i

    # Evaluate the quadrature using the same exponentials as in w_isdf.py
    F_num = -2 * z * np.dot(w, np.exp(-(z * (E_c - E_v) - 2) * tau))

    # Analytic integral of exp(-(z*(E_c-E_v)-1)*t) from 0..inf
    F_exact = -2 * z / (z * (E_c - E_v) - 1)

    assert np.isclose(F_num, F_exact, rtol=1e-6, atol=1e-8)

