import numpy as np


class RebinMixin:
    """Project the histogram onto a new grid by mapping each bin's centroid."""

    def rebin_to(self, target_edges, *, as_storage=None):
        """
        Reproject mass onto target_edges by mapping each source bin's centre
        to the nearest target bin.

        Parameters
        ----------
        target_edges : list of ndarray
        as_storage : storage backend, optional
            If provided, also accumulate results into this backend.
            Useful when rebinning into an existing grid in-place.

        Returns
        -------
        dict  {tuple_index: float}
            Mass in target-grid coordinates.
        """
        src_edges = self.get_edges()
        result = {}

        for idx, count in self.get_mass().items():
            center = [
                (src_edges[d][i] + src_edges[d][i + 1]) / 2.0
                for d, i in enumerate(idx)
            ]

            new_idx = tuple(
                int(np.clip(np.searchsorted(target_edges[d], center[d], side="right") - 1,
                            0, len(target_edges[d]) - 2))
                for d in range(self.dim)
            )

            result[new_idx] = result.get(new_idx, 0.0) + count

        if as_storage is not None:
            for key, val in result.items():
                as_storage.add(key, val)

        return result
