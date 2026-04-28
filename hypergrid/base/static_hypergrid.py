from hypergrid.base._base_tensor import BaseTensorHypergrid
from hypergrid.storage.storage import DictStorage


class StaticHypergrid(BaseTensorHypergrid):
    """
    Hypergrid with user-specified edges and a pluggable storage backend.

    The default storage is DictStorage (sparse, no bounds checking).
    Pass any object that implements add / items / clear / scale to swap backends.

    This is the most flexible variant — useful when you want to combine
    custom storage with the full mixin stack (rebin, compare, visualize).

    Parameters
    ----------
    edges : list of array-like
        Bin edges per dimension.
    storage : storage backend, optional
        Defaults to DictStorage.
    """

    def __init__(self, edges, storage=None):
        super().__init__(edges)
        self.storage = storage or DictStorage()

    def get_mass(self):
        return dict(self.storage.items())
