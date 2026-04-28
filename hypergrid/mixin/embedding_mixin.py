import numpy as np


class EmbeddingMixin:
    """Convert hypergrid mass to a flat, index-ordered probability vector."""

    def to_vector(self):
        """
        Flatten the histogram into a 1D probability vector ordered by
        ravel_multi_index, so two grids with the same edges produce
        directly comparable vectors.

        Returns
        -------
        ndarray, shape (prod(shape),)  — sums to ~1.
        """
        n = int(np.prod(self.shape))
        vec = np.zeros(n)
        for idx, v in self.get_mass().items():
            vec[int(np.ravel_multi_index(idx, self.shape))] = v
        total = vec.sum()
        return vec / (total + 1e-12)
