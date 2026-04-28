import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog


class ComparisonMixin:
    """Statistical divergence metrics between two hypergrids."""

    def compare(self, other, method="js", align="union", normalize=True):
        """
        Compute a scalar distance between self and other.

        Parameters
        ----------
        other : hypergrid
        method : {"l1", "kl", "js", "wasserstein"}
            l1          — total variation distance
            kl          — Kullback-Leibler divergence (asymmetric)
            js          — Jensen-Shannon divergence (symmetric, bounded in [0,1])
            wasserstein — Earth Mover's Distance via linear programming
        align : {"union", "self"}
            union — merge both edge sets before comparing (recommended)
            self  — project other onto self's edges
        normalize : bool
            Compare probability densities (True) or raw counts (False).

        Returns
        -------
        float
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: self has {self.dim}D, other has {other.dim}D.")

        if method == "wasserstein":
            return self._wasserstein(other, normalize)

        h1, h2 = self._align_histograms(other, align)

        if normalize:
            h1 = _normalize(h1)
            h2 = _normalize(h2)

        if method == "l1":
            return _l1(h1, h2)
        elif method == "kl":
            return _kl(h1, h2)
        elif method == "js":
            return _js(h1, h2)
        else:
            raise ValueError(f"Unknown method: {method!r}. Choose from 'l1', 'kl', 'js', 'wasserstein'.")

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _align_histograms(self, other, align):
        if align == "self":
            return self.get_mass(), other.rebin_to(self.get_edges())
        elif align == "union":
            edges = self._union_edges(other)
            return self.rebin_to(edges), other.rebin_to(edges)
        else:
            raise ValueError("align must be 'self' or 'union'.")

    def _union_edges(self, other):
        return [
            np.sort(np.unique(np.concatenate([self.get_edges()[d], other.get_edges()[d]])))
            for d in range(self.dim)
        ]

    # ------------------------------------------------------------------
    # Wasserstein via LP
    # ------------------------------------------------------------------

    def _wasserstein(self, other, normalize=True):
        coords1, w1 = self._to_bin_centers()
        coords2, w2 = other._to_bin_centers()

        if len(coords1) == 0 or len(coords2) == 0:
            return 0.0

        if normalize:
            w1 = w1 / w1.sum()
            w2 = w2 / w2.sum()

        C = cdist(coords1, coords2)
        n, m = len(w1), len(w2)

        # Vectorised constraint matrices (avoids Python loop over n*m).
        # Row constraints: each source bin must ship its full supply w1[i].
        rows = np.eye(n).repeat(m, axis=1)   # (n, n*m)
        # Column constraints: each target bin must receive its full demand w2[j].
        cols = np.tile(np.eye(m), (1, n))     # (m, n*m)

        res = linprog(
            C.flatten(),
            A_eq=np.vstack([rows, cols]),
            b_eq=np.concatenate([w1, w2]),
            bounds=(0, None),
            method="highs",
        )
        return float(res.fun)

    def _to_bin_centers(self):
        edges = self.get_edges()
        coords, weights = [], []
        for idx, count in self.get_mass().items():
            center = [(edges[d][idx[d]] + edges[d][idx[d] + 1]) / 2.0 for d in range(self.dim)]
            coords.append(center)
            weights.append(count)
        return np.array(coords, dtype=float), np.array(weights, dtype=float)


# ------------------------------------------------------------------
# Stateless metric functions — exposed at module level so TemporalHypergrid
# can use them without creating a full hypergrid.
# ------------------------------------------------------------------

def _normalize(hist):
    total = sum(hist.values())
    if total == 0:
        return hist
    return {k: v / total for k, v in hist.items()}


def _l1(h1, h2):
    return sum(abs(h1.get(k, 0.0) - h2.get(k, 0.0)) for k in set(h1) | set(h2))


def _kl(p, q, eps=1e-12):
    return sum(
        p.get(k, 0.0) * np.log((p.get(k, 0.0) + eps) / (q.get(k, 0.0) + eps))
        for k in set(p) | set(q)
    )


def _js(p, q):
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
