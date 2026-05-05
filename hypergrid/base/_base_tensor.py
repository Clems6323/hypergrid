import numpy as np

from hypergrid.base.base_hypergrid import BaseHypergrid
from hypergrid.mixin.rebin_mixin import RebinMixin
from hypergrid.mixin.comparison_mixin import ComparisonMixin
from hypergrid.mixin.embedding_mixin import EmbeddingMixin
from hypergrid.mixin.visualization_mixin import VisualizationMixin
from hypergrid.mixin.stats_mixin import StatsMixin


class BaseTensorHypergrid(
    BaseHypergrid,
    RebinMixin,
    ComparisonMixin,
    EmbeddingMixin,
    VisualizationMixin,
    StatsMixin,
):
    """
    Shared base for all fixed-edge hypergrid variants.

    Provides: _get_bin_index, fit, update, get_edges.
    Subclasses must set self.storage in their __init__ (after super().__init__)
    and implement get_mass().

    Parameters
    ----------
    edges : list of array-like or None
        Bin edges per dimension. When None, subclasses (e.g. AdaptiveHypergrid)
        are responsible for setting self.edges, self.dim, self.shape later.
    """

    def __init__(self, edges=None):
        if edges is not None:
            edges = [np.asarray(e, dtype=float) for e in edges]
            super().__init__(dim=len(edges))
            self.edges = edges
            self.shape = [len(e) - 1 for e in edges]
        else:
            super().__init__(dim=None)
            self.edges = None
            self.shape = None

    def fit(self, data, weights=None):
        self.storage.clear()
        self.update(data, weights)

    def update(self, data, weights=None):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if weights is None:
            weights = np.ones(len(data))
        else:
            weights = np.asarray(weights, dtype=float)

        for point, w in zip(data, weights):
            idx = self._get_bin_index(point)
            if idx is not None:
                self.storage.add(idx, w)

    def _get_bin_index(self, point):
        idx = []
        for d in range(self.dim):
            e = self.edges[d]
            val = point[d]
            if val < e[0] or val >= e[-1]:
                return None
            idx.append(int(np.searchsorted(e, val, side="right")) - 1)
        return tuple(idx)

    def get_edges(self):
        return self.edges
