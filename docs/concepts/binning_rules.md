# Binning rules

`compute_edges` computes bin edges independently per dimension using one of three rules.
Let $n$ be the number of samples, $[lo, hi]$ the data range, and $IQR$ the interquartile range.

## Freedman-Diaconis (`fd`, default)

$$h = 2 \cdot IQR \cdot n^{-1/3}, \quad k = \left\lceil \frac{hi - lo}{h} \right\rceil$$

Robust to outliers. Recommended for most datasets. Falls back to $\sqrt{n}$ when $IQR = 0$.

## Sturges (`sturges`)

$$k = \left\lceil \log_2(n) + 1 \right\rceil$$

Assumes near-Gaussian data. Gives very few bins for large $n$ — best for small samples ($n < 200$).

## Square root (`sqrt`)

$$k = \left\lceil \sqrt{n} \right\rceil$$

Fast heuristic, no distribution assumption. A reasonable default when data is uniform or unknown.

---

All methods cap $k$ at `max_bins` (default 200) to prevent excessively fine grids on large datasets,
and ensure at least 2 bins. Constant dimensions (where $lo = hi$) produce a single bin
$[lo - 0.5, hi + 0.5]$ rather than failing.
