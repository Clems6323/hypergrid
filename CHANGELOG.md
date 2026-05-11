# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-05-11

First stable release.

### Added

#### Grid classes

- `DenseHypergrid` — N-dimensional histogram backed by a dense NumPy array. Efficient for low-dimensional, mostly populated grids.
- `SparseHypergrid` — same interface, sparse dict backend. Efficient for high-dimensional or sparse data.
- `StaticHypergrid` — grid with a pluggable storage backend; accepts any `DictStorage`, `DenseTensorStorage`, or `SparseTensorStorage` instance.
- `AdaptiveHypergrid` — automatically rebins when too many data points fall outside the current boundaries. Replays the full weighted buffer on rebin so no mass is lost.
- `TemporalHypergrid` — wraps any grid to add optional exponential decay of old counts and periodic distribution snapshots for drift analysis.

#### Core workflow

- `fit(data, weights=None)` — build a histogram from scratch; optional per-sample weights.
- `update(data, weights=None)` — accumulate new data into an existing grid without clearing.
- `get_mass()` — return occupied bins as `{tuple_index: float}`.
- `get_edges()` — return the list of edge arrays defining the grid.
- `compute_edges(data, method="fd")` — auto-compute bin edges using Freedman-Diaconis (`"fd"`), Sturges (`"sturges"`), or square-root (`"sqrt"`) rules.

#### Rebinning

- `rebin_to(new_edges)` — project the histogram mass onto a new edge set via centroid mapping; returns a plain `dict`. Mass is conserved.

#### Statistical comparison

- `compare(other, method)` — scalar divergence between two grids. Supported methods: `"l1"` (total variation), `"kl"` (KL divergence), `"js"` (Jensen-Shannon), `"wasserstein"` (Earth Mover's Distance via linear programming).

#### Summary statistics (`describe()`)

- `describe(percentiles=None)` — returns a `pandas.DataFrame` (one column per dimension) with:
  - `count` — total mass
  - `mean` / `std` — probability-weighted from bin centres
  - `skewness` — third standardised central moment
  - `kurtosis` — excess kurtosis (fourth moment / σ⁴ − 3)
  - `min` / `max` — edges of the outermost non-empty bin
  - Configurable percentiles (default 25 %, 50 %, 75 %) via marginal CDF interpolation
- pandas is an optional dependency (`pip install "pyHypergrid[stats]"`).

#### Embedding

- `to_dense()` — export the histogram as a dense NumPy array.
- `to_vector()` — flatten to a normalised probability vector for ML pipelines.
- `sample(n)` — draw `n` points uniformly at random from occupied bins.

#### Visualisation

- `plot_all_marginals()` — one histogram panel per dimension.
- `plot_joint(dim_x, dim_y)` — 2-D joint density heatmap.
- `plot_top_bins(k)` — bar chart of the `k` most populated bins.
- `plot_umap(n_samples, **kwargs)` — UMAP projection of sampled points (requires `umap-learn`).
- `compare_marginal(other, dim, rebin=True)` — overlay marginal histograms of two grids; `rebin=False` plots on their native edges.
- `compare_umap(other, **kwargs)` — UMAP projection coloured by grid membership.

#### Temporal analysis (`TemporalHypergrid`)

- `evolution(method)` — list of pairwise divergences between consecutive snapshots. Supports all four distance methods including Wasserstein.
- `plot_evolution(method)` — line plot of distribution drift over time.
- `plot_temporal_umap(**kwargs)` — UMAP projection of all snapshots coloured by snapshot index.

#### Storage backends

- `DictStorage` — sparse Python dict; general purpose.
- `DenseTensorStorage` — dense NumPy array; fast for low-dimensional grids.
- `SparseTensorStorage` — bounds-checked sparse dict.
- All backends expose a `scale(factor)` method used by `TemporalHypergrid` for exponential decay.

#### Infrastructure

- Full pytest suite covering all grid classes, backends, mixins, and edge cases.
- MkDocs Material documentation with API reference (auto-generated from NumPy docstrings), Concepts page (distances, rebinning, binning rules with LaTeX formulas), and an annotated example notebook.
- GitHub Actions CI: tests across Python 3.10–3.13 on every push and pull request; automatic PyPI publish on version tags via OIDC trusted publishing.
- Issue templates (bug report, feature request) and pull request template.

[1.0.0]: https://github.com/Clems6323/hypergrid/releases/tag/v1.0.0
