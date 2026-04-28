from collections import defaultdict
import numpy as np


class DictStorage:
    """Sparse storage backed by a defaultdict. Zero memory for empty bins."""

    def __init__(self):
        self.data = defaultdict(float)

    def add(self, idx, value):
        self.data[idx] += value

    def get(self, idx, default=0.0):
        return self.data.get(idx, default)

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def scale(self, factor):
        for k in self.data:
            self.data[k] *= factor

    def clear(self):
        self.data.clear()

    def empty_like(self):
        return DictStorage()

    def to_dict(self):
        return dict(self.data)


class DenseTensorStorage:
    """Dense storage backed by a numpy array. O(prod(shape)) memory."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.data = np.zeros(self.shape, dtype=float)

    def add(self, idx, value):
        self.data[idx] += value

    def get(self, idx, default=0.0):
        try:
            return float(self.data[idx])
        except IndexError:
            return default

    def items(self):
        for coord in np.argwhere(self.data != 0):
            yield tuple(coord), float(self.data[tuple(coord)])

    def values(self):
        return (float(self.data[tuple(c)]) for c in np.argwhere(self.data != 0))

    def scale(self, factor):
        self.data *= factor

    def clear(self):
        self.data.fill(0.0)

    def empty_like(self, shape=None):
        return DenseTensorStorage(shape or self.shape)

    def to_dict(self):
        return {idx: v for idx, v in self.items()}


class SparseTensorStorage:
    """Sparse storage with explicit index bounds checking."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.data = {}

    def add(self, idx, value):
        for i, s in zip(idx, self.shape):
            if not (0 <= i < s):
                raise IndexError(f"Index {idx} out of bounds for shape {self.shape}")
        self.data[idx] = self.data.get(idx, 0.0) + value

    def get(self, idx, default=0.0):
        return self.data.get(idx, default)

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def scale(self, factor):
        for k in self.data:
            self.data[k] *= factor

    def clear(self):
        self.data.clear()

    def empty_like(self, shape=None):
        return SparseTensorStorage(shape or self.shape)

    def to_dict(self):
        return dict(self.data)

    def to_dense(self):
        arr = np.zeros(self.shape)
        for idx, v in self.data.items():
            arr[idx] = v
        return arr
