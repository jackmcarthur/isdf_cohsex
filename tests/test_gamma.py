import types
import sys
import os
import numpy as np

# Set up repository root on module path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Minimal CuPy stub to satisfy gpu_utils import
cupy_stub = types.ModuleType('cupy')
for name in ['exp', 'sqrt', 'linspace', 'asarray', 'sum', 'zeros', 'arange']:
    setattr(cupy_stub, name, getattr(np, name))
cupy_stub.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 0))
cupy_stub.complex128 = np.complex128
sys.modules.setdefault('cupy', cupy_stub)

from gamma_matrices import gamma0, gamma1, gamma2, gamma3, gamma5


def test_gamma_anticommutation():
    gammas = [gamma0, gamma1, gamma2, gamma3]
    metric = np.diag([1, -1, -1, -1])
    for i, gi in enumerate(gammas):
        for j, gj in enumerate(gammas):
            lhs = gi @ gj + gj @ gi
            expected = 2 * metric[i, j] * np.eye(4, dtype=np.complex128)
            assert np.allclose(lhs, expected)


def test_gamma5_properties():
    gammas = [gamma0, gamma1, gamma2, gamma3]
    for g in gammas:
        anticom = g @ gamma5 + gamma5 @ g
        assert np.allclose(anticom, np.zeros((4, 4), dtype=np.complex128))
    assert np.allclose(gamma5 @ gamma5, np.eye(4, dtype=np.complex128))
