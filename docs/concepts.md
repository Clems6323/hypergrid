# Concepts

## Distance metrics

Hypergrid supports four metrics for comparing two histograms $P$ and $Q$,
all treating the histograms as discrete probability distributions over bins.
Let $\mathcal{B}$ be the set of all bins (union of both grids' non-empty bins).

---

### L1 — Total variation distance

$$d_{L1}(P, Q) = \sum_{b \in \mathcal{B}} |P(b) - Q(b)|$$

**Range:** $[0, 2]$ (normalised to $[0, 1]$ when $\sum P = \sum Q = 1$)

**Properties:**
- Symmetric: $d_{L1}(P, Q) = d_{L1}(Q, P)$
- Zero iff $P = Q$
- The simplest and cheapest to compute
- Sensitive to absolute differences — a single large bin mismatch dominates

**When to use:** Fast, interpretable first check. Good for detecting gross shifts where many bins change simultaneously.

---

### KL — Kullback-Leibler divergence

$$d_{KL}(P \| Q) = \sum_{b \in \mathcal{B}} P(b) \log \frac{P(b)}{Q(b)}$$

A small $\varepsilon = 10^{-12}$ is added to both $P(b)$ and $Q(b)$ to avoid $\log(0)$.

**Range:** $[0, +\infty)$

**Properties:**
- **Asymmetric**: $d_{KL}(P \| Q) \neq d_{KL}(Q \| P)$ in general
- Penalises heavily when $P(b) > 0$ but $Q(b) \approx 0$
- Unbounded — large values indicate high relative surprise

**When to use:** When you care specifically about how surprising $Q$ is under the model $P$ (e.g. production vs reference). Use with `compare(method="kl")`, keeping in mind which direction matters.

---

### JS — Jensen-Shannon divergence

$$d_{JS}(P, Q) = \frac{1}{2} d_{KL}(P \| M) + \frac{1}{2} d_{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

**Range:** $[0, 1]$ (using natural log; $[0, \log 2]$ using log base 2)

**Properties:**
- Symmetric
- Bounded — easy to compare across experiments
- Smooth: defined even when supports differ
- $\sqrt{d_{JS}}$ is a proper metric (Jensen-Shannon distance)

**When to use:** The recommended default for drift detection and pairwise comparison. Robust, symmetric, and bounded.

---

### Wasserstein — Earth Mover's Distance

The Wasserstein-1 distance (EMD) is the minimum cost to transform $P$ into $Q$,
where cost is the Euclidean distance between bin centres:

$$d_W(P, Q) = \min_{\gamma \geq 0} \sum_{i,j} \gamma_{ij} \cdot \|c_i - c_j\|$$

subject to $\sum_j \gamma_{ij} = P(b_i)$ and $\sum_i \gamma_{ij} = Q(b_j)$,
solved as a linear programme via SciPy `linprog` (HiGHS backend).

**Range:** $[0, +\infty)$, in the same units as the data

**Properties:**
- Symmetric
- Geometrically aware — captures the *distance* mass must travel, not just the amount that differs
- Expensive: $O(n^2)$ LP for $n$ non-empty bins — use small grids or sparse data

**When to use:** When the geometric structure of the shift matters (e.g. a distribution that moved 2 units is more different than one that merely changed shape). Avoid on large dense grids.

---

### Summary

| Metric | Symmetric | Bounded | Geometry-aware | Cost |
|---|:---:|:---:|:---:|---|
| L1 | Yes | Yes ([0,2]) | No | O(B) |
| KL | No | No | No | O(B) |
| JS | Yes | Yes ([0,1]) | No | O(B) |
| Wasserstein | Yes | No | **Yes** | O(B²) LP |

$B$ = number of non-empty bins in the union of both grids.

---

## Rebinning strategy

`rebin_to(target_edges)` reprojects a histogram from its original grid onto a new set of edges,
preserving the total mass as closely as possible.

### Algorithm

For each non-empty source bin $b$ with index $i$ and mass $m_i$:

1. **Compute the bin centre** in data space:
$$c_d = \frac{e_d[i_d] + e_d[i_d+1]}{2} \quad \text{for each dimension } d$$

2. **Find the target bin** that contains $c$:
$$j_d = \text{searchsorted}(\text{target\_edges}_d, c_d) - 1$$
Clipped to $[0, n_\text{bins} - 1]$ so boundary centroids never fall out of range.

3. **Accumulate** mass into target bin $j$:
$$Q(j) \mathrel{+}= m_i$$

### Properties

- **Mass conservation**: $\sum Q = \sum P$ exactly (every source bin's mass lands somewhere in the target)
- **Many-to-one mapping**: multiple source bins can map to the same target bin (coarser grid); each source bin maps to exactly one target bin
- **Loss-free coarsening**: going from fine to coarse loses no mass; going from coarse to fine produces block-uniform distributions (centroids land in a single fine bin)
- **Returns a `dict`** `{tuple_index: float}` — not a grid object — so it can be used with any downstream code without coupling to a specific backend

### Limitations

The centroid-mapping approach is a heuristic: it does not account for the shape of the distribution within each bin. For grids with very different resolutions, consider fitting a new grid directly from raw data rather than rebinning.

---

## Binning rules

`compute_edges` computes bin edges independently per dimension using one of three rules.
Let $n$ be the number of samples, $[lo, hi]$ the data range, and $IQR$ the interquartile range.

### Freedman-Diaconis (`fd`, default)

$$h = 2 \cdot IQR \cdot n^{-1/3}, \quad k = \left\lceil \frac{hi - lo}{h} \right\rceil$$

Robust to outliers. Recommended for most datasets. Falls back to $\sqrt{n}$ when $IQR = 0$.

### Sturges (`sturges`)

$$k = \left\lceil \log_2(n) + 1 \right\rceil$$

Assumes near-Gaussian data. Gives very few bins for large $n$ — best for small samples ($n < 200$).

### Square root (`sqrt`)

$$k = \left\lceil \sqrt{n} \right\rceil$$

Fast heuristic, no distribution assumption. A reasonable default when data is uniform or unknown.

---

All methods cap $k$ at `max_bins` (default 200) to prevent excessively fine grids on large datasets,
and ensure at least 2 bins. Constant dimensions (where $lo = hi$) produce a single bin
$[lo - 0.5, hi + 0.5]$ rather than failing.

---

## Temporal decay

`TemporalHypergrid` applies exponential decay before each `update()` call:

$$H_t \leftarrow \lambda \cdot H_{t-1} + \Delta_t$$

where $\lambda \in (0, 1]$ is the `decay` factor and $\Delta_t$ is the new data batch.

This is equivalent to an exponential moving average over batches, giving more weight to recent data.
The effective half-life in batches is $\log(0.5) / \log(\lambda)$ — e.g. with $\lambda = 0.99$ and
batches of 1000 points, the half-life is ~69 batches (69 000 points).

Snapshots are saved every `snapshot_interval` points and compared with `evolution(method)` to track drift over time.
