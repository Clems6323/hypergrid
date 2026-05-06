import numpy as np


def _pct_label(p):
    return f"{p * 100:.4g}%"


class StatsMixin:
    """Summary statistics computed from the histogram's marginal distributions."""

    def describe(self, percentiles=None):
        """
        Summary statistics of each dimension computed from the binned histogram.

        Statistics are derived from the marginal distribution along each
        dimension (all other dimensions summed out). They reflect the binned
        representation, not the original raw data.

        Parameters
        ----------
        percentiles : list of float, optional
            Quantile positions to include, each in [0, 1].
            Default is [0.25, 0.50, 0.75].

        Returns
        -------
        pandas.DataFrame
            Rows: count, mean, std, skewness, kurtosis, min, <percentiles>, max.
            Columns: integer dimension indices 0, 1, …, dim-1.

        Notes
        -----
        Requires pandas (``pip install pandas``).

        - **count** — total mass (sum of all bin counts, equal across dimensions).
        - **mean / std** — probability-weighted mean and population std of bin centres.
        - **skewness** — third standardised central moment (0 for symmetric distributions).
        - **kurtosis** — fourth standardised central moment minus 3 (excess kurtosis;
          0 for a normal distribution).
        - **min / max** — lower / upper edge of the outermost non-empty bin.
        - **percentiles** — linearly interpolated within bins from the marginal CDF.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "describe() requires pandas.  Install it with: pip install pandas"
            ) from exc

        if percentiles is None:
            percentiles = [0.25, 0.50, 0.75]
        percentiles = sorted(float(p) for p in percentiles)

        edges = self.get_edges()
        mass = self.get_mass()
        columns = {}

        for d in range(self.dim):
            e = np.asarray(edges[d])
            n_bins = len(e) - 1
            centers = (e[:-1] + e[1:]) / 2

            marg = np.zeros(n_bins)
            for idx, v in mass.items():
                marg[idx[d]] += v

            total = marg.sum()

            if total == 0:
                pct_rows = {_pct_label(p): np.nan for p in percentiles}
                columns[d] = {
                    "count": 0.0, "mean": np.nan, "std": np.nan,
                    "skewness": np.nan, "kurtosis": np.nan,
                    "min": np.nan, **pct_rows, "max": np.nan,
                }
                continue

            w = marg / total
            mean = float(np.dot(w, centers))
            deviations = centers - mean
            variance = float(np.dot(w, deviations ** 2))
            std = float(np.sqrt(variance))

            # Standardised central moments; guard against zero-variance (single bin).
            if std > 0:
                skewness = float(np.dot(w, deviations ** 3) / std ** 3)
                kurtosis = float(np.dot(w, deviations ** 4) / std ** 4) - 3.0
            else:
                skewness = 0.0
                kurtosis = 0.0

            nonempty = np.where(marg > 0)[0]
            min_val = float(e[nonempty[0]])
            max_val = float(e[nonempty[-1] + 1])

            # Percentiles: linear interpolation within bins from the marginal CDF.
            cdf = np.cumsum(marg) / total
            pct_rows = {}
            for p in percentiles:
                i = int(np.searchsorted(cdf, p, side="left"))
                i = min(i, n_bins - 1)
                cdf_lo = cdf[i - 1] if i > 0 else 0.0
                cdf_hi = cdf[i]
                t = (p - cdf_lo) / (cdf_hi - cdf_lo) if cdf_hi > cdf_lo else 0.5
                pct_rows[_pct_label(p)] = float(e[i] + t * (e[i + 1] - e[i]))

            columns[d] = {
                "count": float(total),
                "mean": mean,
                "std": std,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "min": min_val,
                **pct_rows,
                "max": max_val,
            }

        return pd.DataFrame(columns)
