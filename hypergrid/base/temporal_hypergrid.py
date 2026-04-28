import numpy as np
import matplotlib.pyplot as plt

from hypergrid.mixin.comparison_mixin import _js, _kl, _l1, _normalize


class TemporalHypergrid:
    """
    Wrapper that adds temporal tracking to any hypergrid.

    Applies optional exponential decay to old counts before each update and
    saves periodic snapshots of the distribution so drift can be measured.

    Parameters
    ----------
    grid : BaseTensorHypergrid instance
        The underlying hypergrid (DenseTensorHypergrid, SparseHypergrid, etc.)
    decay : float, optional
        Multiplicative factor applied to all counts before each update batch
        (e.g. 0.99 for slow forgetting). None means no decay.
    snapshot_interval : int
        Save a snapshot every N data points processed.
    """

    def __init__(self, grid, decay=None, snapshot_interval=1000):
        self.grid = grid
        self.decay = decay
        self.snapshot_interval = snapshot_interval
        self._counter = 0
        self.snapshots = []

    # ------------------------------------------------------------------
    # Forwarded interface — TemporalHypergrid can be used wherever a
    # plain hypergrid is expected.
    # ------------------------------------------------------------------

    @property
    def dim(self):
        return self.grid.dim

    @property
    def edges(self):
        return self.grid.edges

    @property
    def shape(self):
        return self.grid.shape

    def get_mass(self):
        return self.grid.get_mass()

    def get_edges(self):
        return self.grid.get_edges()

    def fit(self, data, weights=None):
        self.grid.fit(data, weights)
        self._counter = 0
        self.snapshots.clear()

    def update(self, data, weights=None):
        data = np.asarray(data)

        if self.decay is not None:
            self.grid.storage.scale(self.decay)

        self.grid.update(data, weights)
        self._counter += len(data)

        if self.snapshot_interval and (self._counter % self.snapshot_interval == 0):
            self.snapshots.append(dict(self.grid.get_mass()))

    # ------------------------------------------------------------------
    # Temporal analysis
    # ------------------------------------------------------------------

    def evolution(self, method="js"):
        """
        Compute divergence between consecutive snapshots.

        Parameters
        ----------
        method : {"js", "kl", "l1"}

        Returns
        -------
        list of float
        """
        if len(self.snapshots) < 2:
            return []

        fn = {"js": _js, "kl": _kl, "l1": _l1}[method]
        return [
            fn(_normalize(h1), _normalize(h2))
            for h1, h2 in zip(self.snapshots, self.snapshots[1:])
        ]

    def plot_evolution(self, method="js"):
        """Plot divergence between consecutive snapshots over time."""
        distances = self.evolution(method)
        if not distances:
            raise ValueError("Need at least 2 snapshots. Process more data or reduce snapshot_interval.")
        plt.plot(distances, marker="o")
        plt.title(f"Distribution drift over time ({method.upper()})")
        plt.xlabel("Snapshot index")
        plt.ylabel("Divergence")
        plt.tight_layout()
        plt.show()

    def plot_temporal_umap(self, n_per_snapshot=500, **umap_kwargs):
        """
        UMAP projection of all snapshots coloured by snapshot index.
        Each snapshot contributes n_per_snapshot sampled points.
        """
        import umap as _umap
        from hypergrid.base.sparse_tensor_hypergrid import SparseTensorHypergrid

        samples, labels = [], []
        for i, snap in enumerate(self.snapshots):
            tmp = SparseTensorHypergrid(self.grid.get_edges())
            tmp.storage.data.update(snap)
            pts = tmp.sample(n_per_snapshot)
            samples.append(pts)
            labels.extend([i] * len(pts))

        X = np.vstack(samples)
        emb = _umap.UMAP(**umap_kwargs).fit_transform(X)

        plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=3, cmap="viridis")
        plt.colorbar(label="Snapshot index")
        plt.title("Temporal evolution (UMAP)")
        plt.tight_layout()
        plt.show()
