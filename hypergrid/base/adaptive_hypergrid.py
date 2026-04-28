import numpy as np
from collections import deque

from hypergrid.base._base_tensor import BaseTensorHypergrid
from hypergrid.storage.storage import DictStorage
from hypergrid.utils.binning import compute_edges


class AdaptiveHypergrid(BaseTensorHypergrid):
    """
    Hypergrid that automatically rebins when the overflow fraction exceeds
    a threshold, adapting the grid to concept drift.

    A rolling buffer of recent points is kept so that rebinning uses the
    most recent data distribution rather than the full history.

    Parameters
    ----------
    edges : list of array-like, optional
        Initial bin edges. If None, computed from the first batch passed to fit.
    drift_threshold : float
        Fraction of out-of-bounds points that triggers a rebin (default 0.05).
    buffer_size : int
        Maximum number of recent points kept for rebinning.
    binning_method : {"fd", "sturges", "sqrt"}
        Edge computation method used during rebinning.
    max_bins : int
        Per-dimension bin cap used during rebinning.
    """

    def __init__(
        self,
        edges=None,
        drift_threshold=0.05,
        buffer_size=5000,
        binning_method="fd",
        max_bins=50,
    ):
        super().__init__(edges)  # edges=None is handled by BaseTensorHypergrid
        self.storage = DictStorage()
        self.buffer = deque(maxlen=buffer_size)
        self.drift_threshold = drift_threshold
        self.binning_method = binning_method
        self.max_bins = max_bins

        self._overflow = 0.0
        self._total = 0.0
        self._drift_history = []  # overflow fraction recorded at each rebin

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def fit(self, data, weights=None):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        self._init_edges(data)
        self.storage.clear()
        self._overflow = 0.0
        self._total = 0.0
        self.buffer.clear()
        self._accumulate(data, weights)

    def update(self, data, weights=None):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[np.newaxis, :]

        if self.edges is None:
            # Auto-initialise on first update if fit() was never called.
            self._init_edges(data)

        self._accumulate(data, weights)

    def get_mass(self):
        return dict(self.storage.items())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _accumulate(self, data, weights):
        if weights is None:
            weights = np.ones(len(data))
        else:
            weights = np.asarray(weights, dtype=float)

        for point, w in zip(data, weights):
            self.buffer.append(point)
            self._total += w
            idx = self._get_bin_index(point)
            if idx is None:
                self._overflow += w
            else:
                self.storage.add(idx, w)

        if self._total > 0 and (self._overflow / self._total) > self.drift_threshold:
            self._rebin()

    def _rebin(self):
        self._drift_history.append(self._overflow / self._total)

        data = np.array(self.buffer)
        self._init_edges(data)
        self.storage.clear()
        self._overflow = 0.0
        self._total = 0.0

        # Re-bin buffered points without triggering another rebin (direct add).
        for point in data:
            idx = self._get_bin_index(point)
            if idx is not None:
                self.storage.add(idx, 1.0)
            else:
                self._overflow += 1.0
            self._total += 1.0

    def _init_edges(self, data):
        edges = compute_edges(data, method=self.binning_method, max_bins=self.max_bins)
        self.edges = [np.asarray(e, dtype=float) for e in edges]
        self.dim = len(self.edges)
        self.shape = [len(e) - 1 for e in self.edges]
