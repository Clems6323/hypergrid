import numpy as np

from hypergrid.base._base_tensor import BaseTensorHypergrid
from hypergrid.storage.storage import SparseTensorStorage


class SparseTensorHypergrid(BaseTensorHypergrid):
    """
    Hypergrid backed by a sparse dict with explicit bounds checking.

    Best for high-dimensional grids or data that occupies only a small
    fraction of the bin space. Memory scales with non-zero bin count.

    Parameters
    ----------
    edges : list of array-like
        Bin edges per dimension (length n_bins_d + 1 each).
    """

    def __init__(self, edges):
        super().__init__(edges)
        self.storage = SparseTensorStorage(self.shape)

    def get_mass(self):
        return self.storage.to_dict()

    def to_dense(self):
        arr = np.zeros(self.shape)
        for idx, v in self.storage.items():
            arr[idx] = v
        return arr

    def to_sparse(self):
        return self.storage.to_dict()
