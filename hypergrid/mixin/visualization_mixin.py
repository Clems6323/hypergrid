import numpy as np
import matplotlib.pyplot as plt


class VisualizationMixin:
    """Plotting methods for single-grid inspection and pairwise comparison."""

    # ------------------------------------------------------------------
    # Single-grid plots
    # ------------------------------------------------------------------

    def plot_top_bins(self, k=20, ax=None):
        items = sorted(self.get_mass().items(), key=lambda x: -x[1])[:k]
        labels = [str(idx) for idx, _ in items]
        values = [v for _, v in items]

        standalone = ax is None
        ax = ax or plt.gca()
        ax.barh(labels[::-1], values[::-1])
        ax.set_title(f"Top {k} bins by mass")
        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_marginal(self, dim, ax=None, label=None):
        edges = self.get_edges()[dim]
        hist = np.zeros(len(edges) - 1)
        for idx, v in self.get_mass().items():
            hist[idx[dim]] += v
        hist /= hist.sum() + 1e-12

        standalone = ax is None
        ax = ax or plt.gca()
        ax.plot((edges[:-1] + edges[1:]) / 2, hist, marker="o", label=label)
        ax.set_title(f"Marginal (dim {dim})")
        ax.set_xlabel(f"Dimension {dim}")
        ax.set_ylabel("Probability")
        if label:
            ax.legend()
        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_all_marginals(self):
        fig, axes = plt.subplots(1, self.dim, figsize=(5 * self.dim, 4))
        axes = [axes] if self.dim == 1 else list(axes)
        for d, ax in enumerate(axes):
            self.plot_marginal(d, ax=ax)
        plt.tight_layout()
        plt.show()

    def plot_joint(self, dim_x, dim_y, ax=None):
        ex, ey = self.get_edges()[dim_x], self.get_edges()[dim_y]
        grid = np.zeros((len(ex) - 1, len(ey) - 1))
        for idx, v in self.get_mass().items():
            grid[idx[dim_x], idx[dim_y]] += v
        grid /= grid.sum() + 1e-12

        standalone = ax is None
        ax = ax or plt.gca()
        im = ax.imshow(grid.T, origin="lower", aspect="auto")
        plt.colorbar(im, ax=ax, label="Probability")
        ax.set_title(f"Joint ({dim_x}, {dim_y})")
        ax.set_xlabel(f"Dim {dim_x}")
        ax.set_ylabel(f"Dim {dim_y}")
        if standalone:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples=2000, rng=None):
        """
        Draw samples from the histogram by treating each bin as a uniform
        distribution over its volume.

        Parameters
        ----------
        n_samples : int
        rng : numpy.random.Generator, optional

        Returns
        -------
        ndarray, shape (n_samples, dim)
        """
        rng = np.random.default_rng(rng)
        mass = self.get_mass()
        if not mass:
            raise ValueError("Histogram is empty; call fit() or update() first.")

        bins = list(mass.keys())
        weights = np.array(list(mass.values()), dtype=float)
        weights /= weights.sum()

        chosen = rng.choice(len(bins), size=n_samples, p=weights)
        edges = self.get_edges()

        samples = np.empty((n_samples, self.dim))
        for i, c in enumerate(chosen):
            idx = bins[c]
            for d in range(self.dim):
                lo, hi = edges[d][idx[d]], edges[d][idx[d] + 1]
                samples[i, d] = rng.uniform(lo, hi)
        return samples

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------

    def plot_umap(self, n_samples=2000, ax=None, **umap_kwargs):
        """UMAP projection of a single hypergrid."""
        import umap as _umap
        emb = _umap.UMAP(**umap_kwargs).fit_transform(self.sample(n_samples))

        standalone = ax is None
        ax = ax or plt.gca()
        ax.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.6)
        ax.set_title("UMAP projection")
        if standalone:
            plt.tight_layout()
            plt.show()

    def compare_umap(self, other, n_samples=2000, **umap_kwargs):
        """UMAP projection with both grids overlaid in different colours."""
        import umap as _umap
        X1, X2 = self.sample(n_samples), other.sample(n_samples)
        labels = np.array([0] * len(X1) + [1] * len(X2))
        emb = _umap.UMAP(**umap_kwargs).fit_transform(np.vstack([X1, X2]))

        fig, ax = plt.subplots()
        ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1], s=5, alpha=0.6, label="self")
        ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1], s=5, alpha=0.6, label="other")
        ax.legend()
        ax.set_title("UMAP comparison")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------

    def compare_marginal(self, other, dim, ax=None):
        """Overlay marginal distributions for dimension `dim`."""
        edges = self.get_edges()[dim]
        other_proj = other.rebin_to(self.get_edges())

        h1 = np.zeros(len(edges) - 1)
        h2 = np.zeros(len(edges) - 1)
        for idx, v in self.get_mass().items():
            h1[idx[dim]] += v
        for idx, v in other_proj.items():
            h2[idx[dim]] += v

        h1 /= h1.sum() + 1e-12
        h2 /= h2.sum() + 1e-12
        centers = (edges[:-1] + edges[1:]) / 2

        standalone = ax is None
        ax = ax or plt.gca()
        ax.plot(centers, h1, label="self")
        ax.plot(centers, h2, label="other")
        ax.legend()
        ax.set_title(f"Marginal comparison (dim {dim})")
        if standalone:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Drift (AdaptiveHypergrid)
    # ------------------------------------------------------------------

    def plot_drift(self):
        if not hasattr(self, "_drift_history") or not self._drift_history:
            raise ValueError("No drift history available. Use AdaptiveHypergrid and process data first.")
        plt.plot(self._drift_history, marker="o")
        plt.title("Overflow fraction at each rebin event")
        plt.xlabel("Rebin index")
        plt.ylabel("Overflow fraction")
        plt.tight_layout()
        plt.show()
