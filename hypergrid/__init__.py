"""
Hypergrid — n-dimensional histogram library for multivariate data.

Main classes
------------
DenseHypergrid    : fixed-edge grid, dense numpy backend (low-dim, mostly full grids)
SparseHypergrid   : fixed-edge grid, sparse dict backend (high-dim or sparse data)
AdaptiveHypergrid : auto-rebinning when distribution shifts outside the current grid
TemporalHypergrid : wrapper adding exponential decay and snapshot-based drift tracking

Utility
-------
compute_edges : auto-compute bin edges from data (Freedman-Diaconis, Sturges, sqrt)

Advanced
--------
StaticHypergrid    : fixed-edge grid with pluggable storage backend
BaseTensorHypergrid: base class for building custom implementations
"""

from hypergrid.base.dense_tensor_hypergrid import DenseTensorHypergrid as DenseHypergrid
from hypergrid.base.sparse_tensor_hypergrid import SparseTensorHypergrid as SparseHypergrid
from hypergrid.base.static_hypergrid import StaticHypergrid
from hypergrid.base.adaptive_hypergrid import AdaptiveHypergrid
from hypergrid.base.temporal_hypergrid import TemporalHypergrid
from hypergrid.base._base_tensor import BaseTensorHypergrid
from hypergrid.utils.binning import compute_edges

# Original names kept as aliases.
DenseTensorHypergrid = DenseHypergrid
SparseTensorHypergrid = SparseHypergrid

__version__ = "0.1.0"

__all__ = [
    "DenseHypergrid",
    "SparseHypergrid",
    "StaticHypergrid",
    "AdaptiveHypergrid",
    "TemporalHypergrid",
    "BaseTensorHypergrid",
    "compute_edges",
    # aliases
    "DenseTensorHypergrid",
    "SparseTensorHypergrid",
]
