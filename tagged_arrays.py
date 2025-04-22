import numpy as np
import cupy
import cupyx.scipy.fftpack as cufft
from numpy import s_
from typing import Union

class LabeledArray:
    __slots__ = ['data', 'axes', 'xp', 'joined_axes']

    def __init__(self, data=None, axes=None, shape=None, dtype=np.complex128, joined_axes=None):
        if data is not None:
            assert axes is not None and len(axes) == data.ndim
            self.data = data
            self.xp = cupy.get_array_module(data)
            self.axes = list(axes)
        elif axes is not None and shape is not None:
            # interpret None as newaxis (size 1)
            shape_resolved = tuple(1 if dim is None else dim for dim in shape)
            self.xp = cupy if cupy.is_available() else np
            self.data = self.xp.zeros(shape_resolved, dtype=dtype)
            self.axes = list(axes)
        else:
            raise ValueError("Either data+axes or axes+shape must be provided.")
        self.joined_axes = joined_axes or {}

    def transpose(self, *new_order):
        if set(new_order) != set(self.axes):
            raise ValueError("new_order must be a permutation of current axes.")
        perm = [self.axes.index(ax) for ax in new_order]
        new_data = self.xp.ascontiguousarray(self.data.transpose(perm))
        return LabeledArray(new_data, list(new_order), joined_axes=dict(self.joined_axes))

    def slice(self, axis_name: str, slice_val: Union[int, slice]):
        idx = self.axes.index(axis_name)
        slicer = [slice(None)] * self.data.ndim
        slicer[idx] = slice_val
        new_data = self.data[tuple(slicer)]
        new_axes = self.axes[:idx] + self.axes[idx+1:] if isinstance(slice_val, int) else self.axes
        return LabeledArray(new_data, new_axes, joined_axes=dict(self.joined_axes))

    def slice_many(self, slice_dict: dict):
        slicer = [slice(None)] * self.data.ndim
        new_axes = list(self.axes)
        for ax, s_val in slice_dict.items():
            if ax not in self.axes:
                raise ValueError(f"Axis '{ax}' not found.")
            idx = self.axes.index(ax)
            slicer[idx] = s_val
            if isinstance(s_val, int):
                new_axes[idx] = None  # mark for deletion
        new_axes = [ax for ax in new_axes if ax is not None]
        new_data = self.data[tuple(slicer)]
        return LabeledArray(new_data, new_axes, joined_axes=dict(self.joined_axes))

    def shape_dict(self):
        return {name: self.data.shape[i] for i, name in enumerate(self.axes)}

    def kgrid_to_last(self):
        grid_axes = ['nkx', 'nky', 'nkz']
        front = [a for a in self.axes if a not in grid_axes]
        return self.transpose(*front, *grid_axes)

    def freq_to_last(self):
        if 'nfreq' not in self.axes:
            raise ValueError("nfreq axis not present.")
        front = [a for a in self.axes if a != 'nfreq']
        return self.transpose(*front, 'nfreq')

    def join(self, *axes_to_join):
        axes_to_join = list(axes_to_join)
        idxs = [self.axes.index(ax) for ax in axes_to_join]
        if sorted(idxs) != list(range(min(idxs), max(idxs)+1)):
            raise ValueError("Axes must be contiguous to join.")
        new_dim = int(np.prod([self.data.shape[i] for i in idxs]))
        new_shape = (
            self.data.shape[:idxs[0]] +
            (new_dim,) +
            self.data.shape[idxs[-1]+1:]
        )
        new_axes = (
            self.axes[:idxs[0]] +
            ['*'.join(axes_to_join)] +
            self.axes[idxs[-1]+1:]
        )
        new_data = self.data.reshape(new_shape)
        joined_axes = dict(self.joined_axes)
        joined_axes['*'.join(axes_to_join)] = axes_to_join
        return LabeledArray(new_data, new_axes, joined_axes)

    def unjoin(self, *original_axes):
        joined_name = '*'.join(original_axes)
        if joined_name not in self.joined_axes:
            raise ValueError(f"Axes {original_axes} are not currently joined.")
        idx = self.axes.index(joined_name)
        recovered_axes = self.joined_axes[joined_name]
        shape_dict = {ax: sz for ax, sz in zip(self.axes, self.data.shape)}
        recovered_shapes = [shape_dict[joined_name] // np.prod([shape_dict[a] for a in recovered_axes])] * len(recovered_axes)
        new_shape = (
            self.data.shape[:idx] +
            tuple(recovered_shapes) +
            self.data.shape[idx+1:]
        )
        new_axes = self.axes[:idx] + recovered_axes + self.axes[idx+1:]
        new_data = self.data.reshape(new_shape)
        new_joined = dict(self.joined_axes)
        del new_joined[joined_name]
        return LabeledArray(new_data, new_axes, new_joined)

    def fft_kgrid(self):
        self._apply_fft_inplace(['nkx', 'nky', 'nkz'], inverse=False)

    def ifft_kgrid(self):
        self._apply_fft_inplace(['nkx', 'nky', 'nkz'], inverse=True)

    def _apply_fft_inplace(self, k_axes, inverse=False):
        if self.xp is not cupy:
            raise RuntimeError("fft_kgrid/ifft_kgrid requires CuPy.")

        k_idxs = [self.axes.index(ax) for ax in k_axes]
        plan = cufft.get_fft_plan(self.data, axes=tuple(k_idxs))
        func = cufft.ifftn if inverse else cufft.fftn

        # Do in-place FFT by reassigning to self.data
        self.data[...] = func(self.data, axes=k_idxs, plan=plan, overwrite_x=True)

    def __repr__(self):
        return f"LabeledArray(shape={self.data.shape}, axes={self.axes}, joined_axes={self.joined_axes})"
