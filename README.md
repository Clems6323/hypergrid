# Hypergrid

<p align="center">
  <img src="docs/hypergrid_logo.png" alt="Hypergrid" width="300"/>
</p>

N-dimensional histogram library for multivariate data.

Hypergrid bins arbitrary-dimensional data into a grid, then lets you update it incrementally, reproject it onto new edges, compare two grids statistically, and visualize the distribution — including UMAP projections and temporal drift.

## Installation

```bash
pip install hypergrid
# with UMAP support:
pip install "hypergrid[umap]"
```

## Quick start

```python
import numpy as np
from hypergrid import DenseHypergrid, SparseHypergrid, AdaptiveHypergrid, compute_edges

data = np.random.randn(5000, 3)

# Auto-compute edges with Freedman-Diaconis rule
edges = compute_edges(data)

# Dense backend (good for low-dim, mostly populated grids)
grid = DenseHypergrid(edges)
grid.fit(data)

# Sparse backend (good for high-dim or sparse data)
grid = SparseHypergrid(edges)
grid.fit(data)

# Incremental update
grid.update(np.random.randn(500, 3))
```

## Adaptive grid

Automatically rebins when too much data falls outside the current boundaries:

```python
from hypergrid import AdaptiveHypergrid

grid = AdaptiveHypergrid(drift_threshold=0.05, buffer_size=5000)
grid.fit(data_batch_1)
grid.update(data_batch_2)   # rebins if >5% overflow
```

## Comparing grids

```python
grid1 = DenseHypergrid(edges); grid1.fit(data1)
grid2 = DenseHypergrid(edges); grid2.fit(data2)

grid1.compare(grid2, method="js")           # Jensen-Shannon divergence
grid1.compare(grid2, method="wasserstein")  # Earth Mover's Distance
grid1.compare(grid2, method="l1")           # Total variation
grid1.compare(grid2, method="kl")           # KL divergence
```

## Visualization

```python
grid.plot_all_marginals()
grid.plot_joint(dim_x=0, dim_y=1)
grid.plot_top_bins(k=20)

grid.plot_umap(n_samples=3000)
grid1.compare_umap(grid2)
grid1.compare_marginal(grid2, dim=0)
```

## Temporal tracking

```python
from hypergrid import DenseHypergrid, TemporalHypergrid

base = DenseHypergrid(edges)
tgrid = TemporalHypergrid(base, decay=0.99, snapshot_interval=1000)

for batch in stream:
    tgrid.update(batch)

tgrid.plot_evolution(method="js")
tgrid.plot_temporal_umap()
```

## Rebinning

```python
new_edges = compute_edges(new_data)
rebinned = grid.rebin_to(new_edges)   # returns dict {index: count}
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
| `DenseTensorHypergrid` | numpy array | Low-dim, mostly full grids |
| `SparseTensorHypergrid` | sparse dict (bounds-checked) | High-dim or sparse data |
| `StaticHypergrid` | pluggable (default: DictStorage) | Custom backends |

## License

MIT
