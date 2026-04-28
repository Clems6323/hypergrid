import numpy as np

from hypergrid.base._base_tensor import BaseTensorHypergrid
from hypergrid.storage.storage import DenseTensorStorage


class DenseTensorHypergrid(BaseTensorHypergrid):
    """
    Hypergrid backed by a dense numpy array.

    Best for low-dimensional grids where most bins receive data.
    Memory footprint is O(prod(n_bins_per_dim)), independent of sparsity.

    Parameters
    ----------
    edges : list of array-like
        Bin edges per dimension (length n_bins_d + 1 each).
    """

    def __init__(self, edges):
        super().__init__(edges)
        self.storage = DenseTensorStorage(self.shape)

    def get_mass(self):
        return self.storage.to_dict()

    def to_dense(self):
        return self.storage.data.copy()

    def to_sparse(self):
        return self.storage.to_dict()
