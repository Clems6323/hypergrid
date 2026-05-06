# Distance metrics

Hypergrid supports four metrics for comparing two histograms $P$ and $Q$,
all treating the histograms as discrete probability distributions over bins.
Let $\mathcal{B}$ be the set of all bins (union of both grids' non-empty bins).

---

## L1 — Total variation distance

$$d_{L1}(P, Q) = \sum_{b \in \mathcal{B}} |P(b) - Q(b)|$$

**Range:** $[0, 2]$ (normalised to $[0, 1]$ when $\sum P = \sum Q = 1$)

**Properties:**
- Symmetric: $d_{L1}(P, Q) = d_{L1}(Q, P)$
- Zero iff $P = Q$
- The simplest and cheapest to compute
- Sensitive to absolute differences — a single large bin mismatch dominates

**When to use:** Fast, interpretable first check. Good for detecting gross shifts where many bins change simultaneously.

---

## KL — Kullback-Leibler divergence

$$d_{KL}(P \| Q) = \sum_{b \in \mathcal{B}} P(b) \log \frac{P(b)}{Q(b)}$$

A small $\varepsilon = 10^{-12}$ is added to both $P(b)$ and $Q(b)$ to avoid $\log(0)$.

**Range:** $[0, +\infty)$

**Properties:**
- **Asymmetric**: $d_{KL}(P \| Q) \neq d_{KL}(Q \| P)$ in general
- Penalises heavily when $P(b) > 0$ but $Q(b) \approx 0$
- Unbounded — large values indicate high relative surprise

**When to use:** When you care specifically about how surprising $Q$ is under the model $P$ (e.g. production vs reference). Use with `compare(method="kl")`, keeping in mind which direction matters.

---

## JS — Jensen-Shannon divergence

$$d_{JS}(P, Q) = \frac{1}{2} d_{KL}(P \| M) + \frac{1}{2} d_{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

**Range:** $[0, 1]$ (using natural log; $[0, \log 2]$ using log base 2)

**Properties:**
- Symmetric
- Bounded — easy to compare across experiments
- Smooth: defined even when supports differ
- $\sqrt{d_{JS}}$ is a proper metric (Jensen-Shannon distance)

**When to use:** The recommended default for drift detection and pairwise comparison. Robust, symmetric, and bounded.

---

## Wasserstein — Earth Mover's Distance

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

## Summary

| Metric | Symmetric | Bounded | Geometry-aware | Cost |
|---|:---:|:---:|:---:|---|
| L1 | Yes | Yes ([0,2]) | No | O(B) |
| KL | No | No | No | O(B) |
| JS | Yes | Yes ([0,1]) | No | O(B) |
| Wasserstein | Yes | No | **Yes** | O(B²) LP |

$B$ = number of non-empty bins in the union of both grids.
