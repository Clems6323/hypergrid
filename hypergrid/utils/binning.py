import numpy as np


def compute_bins_1d(x, method="fd", max_bins=200):
    """
    Compute 1D bin edges using a histogram rule.

    Parameters
    ----------
    x : array-like, shape (n,)
    method : {"fd", "sturges", "sqrt"}
        fd      — Freedman-Diaconis (robust to outliers, recommended default)
        sturges — log2(n) + 1  (assumes near-Gaussian, suitable for small n)
        sqrt    — sqrt(n)      (fast heuristic)
    max_bins : int
        Upper cap to prevent excessively fine grids on large datasets.

    Returns
    -------
    edges : ndarray, shape (n_bins + 1,)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    lo, hi = x.min(), x.max()

    if lo == hi:
        # Degenerate dimension: create a single bin around the constant value.
        return np.array([lo - 0.5, hi + 0.5])

    if method == "fd":
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            n_bins = int(np.ceil(np.sqrt(n)))
        else:
            bw = 2.0 * iqr * n ** (-1.0 / 3.0)
            n_bins = int(np.ceil((hi - lo) / bw))
    elif method == "sturges":
        n_bins = int(np.ceil(np.log2(n) + 1))
    elif method == "sqrt":
        n_bins = int(np.ceil(np.sqrt(n)))
    else:
        raise ValueError(f"Unknown binning method: {method!r}. Use 'fd', 'sturges', or 'sqrt'.")

    n_bins = max(2, min(n_bins, max_bins))
    return np.linspace(lo, hi, n_bins + 1)


def compute_edges(data, method="fd", max_bins=200):
    """
    Compute bin edges independently for each dimension of a 2D dataset.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
    method : str
        Passed to compute_bins_1d for each dimension.
    max_bins : int
        Per-dimension cap.

    Returns
    -------
    edges : list of ndarray, length n_features
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return [compute_bins_1d(data[:, d], method=method, max_bins=max_bins)
            for d in range(data.shape[1])]
