# Hypergrid

<p align="center">
  <img src="hypergrid_logo.png" alt="Hypergrid" width="300"/>
</p>

**N-dimensional histogram library for multivariate data.**

Hypergrid bins arbitrary-dimensional data into a grid, then lets you update it incrementally, reproject it onto new edges, compare two grids statistically, and visualize the distribution — including UMAP projections and temporal drift.

## Installation

```bash
# pip
pip install pyHypergrid

# uv
uv add pyHypergrid
```

Optional extras:

```bash
# pip — with pandas (describe()) and/or UMAP support
pip install "pyHypergrid[stats]"
pip install "pyHypergrid[umap]"
pip install "pyHypergrid[stats,umap]"

# uv
uv add "pyHypergrid[stats]"
uv add "pyHypergrid[umap]"
uv add "pyHypergrid[stats,umap]"
```

## Quick start

```python
import numpy as np
from hypergrid import DenseHypergrid, SparseHypergrid, AdaptiveHypergrid, compute_edges

data = np.random.randn(5000, 3)

# Auto-compute edges with Freedman-Diaconis rule
edges = compute_edges(data)

# Dense backend — good for low-dim, mostly populated grids
grid = DenseHypergrid(edges)
grid.fit(data)

# Sparse backend — good for high-dim or sparse data
grid = SparseHypergrid(edges)
grid.fit(data)

# Incremental update
grid.update(np.random.randn(500, 3))
```

## Architecture

```
BaseHypergrid  (ABC)
  └─ BaseTensorHypergrid  (+ RebinMixin, ComparisonMixin, EmbeddingMixin, VisualizationMixin)
       ├─ DenseTensorHypergrid   — numpy array backend
       ├─ SparseTensorHypergrid  — bounds-checked sparse dict
       ├─ StaticHypergrid        — pluggable storage backend
       └─ AdaptiveHypergrid      — auto-rebinning on drift

TemporalHypergrid  — wraps any hypergrid, adds decay + snapshots
```

## Storage backends

| Class | Backend | Best for |
|---|---|---|
| `DenseHypergrid` | numpy array | Low-dim, mostly full grids |
| `SparseHypergrid` | sparse dict (bounds-checked) | High-dim or sparse data |
| `StaticHypergrid` | pluggable (default: DictStorage) | Custom backends |

## Features at a glance

- **Fit & update** — build a histogram from scratch or accumulate new data
- **Rebin** — project mass onto a different edge set
- **Compare** — L1, KL, Jensen-Shannon, Wasserstein distances between grids
- **Embed** — flatten histogram to a probability vector for ML pipelines
- **Visualize** — marginals, joint plots, top bins, UMAP, temporal drift
- **Adaptive** — auto-rebin when data drifts outside current boundaries
- **Temporal** — exponential decay + snapshots for streaming data

See the [API Reference](api.md) for full documentation, or jump into the [example notebooks](examples/01_basic.ipynb).
