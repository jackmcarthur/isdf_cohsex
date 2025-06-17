from gpu_utils import xp

"""Dirac gamma matrices for use with CPU or GPU arrays.

The matrices are provided in the standard Dirac representation
and stored as ``xp`` arrays so code can remain agnostic to
NumPy or CuPy backends.
"""

# Pauli matrices
sigma_x = xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
sigma_y = xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128)
sigma_z = xp.array([[1, 0], [0, -1]], dtype=xp.complex128)

_I2 = xp.eye(2, dtype=xp.complex128)
_Z2 = xp.zeros((2, 2), dtype=xp.complex128)

# Standard Dirac gamma matrices

gamma0 = xp.block([[ _I2, _Z2],
                   [ _Z2, -_I2]])

gamma1 = xp.block([[ _Z2,  sigma_x],
                   [-sigma_x, _Z2]])

gamma2 = xp.block([[ _Z2,  sigma_y],
                   [-sigma_y, _Z2]])

gamma3 = xp.block([[ _Z2,  sigma_z],
                   [-sigma_z, _Z2]])

# gamma^5 = i gamma^0 gamma^1 gamma^2 gamma^3
# In the Dirac basis this reduces to off-diagonal identity blocks

gamma5 = xp.block([[ _Z2,  _I2],
                   [ _I2,  _Z2]])

__all__ = [
    "sigma_x", "sigma_y", "sigma_z",
    "gamma0", "gamma1", "gamma2", "gamma3", "gamma5",
]
