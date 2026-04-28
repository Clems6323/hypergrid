from abc import ABC, abstractmethod


class BaseHypergrid(ABC):
    """
    Abstract interface for all Hypergrid implementations.

    Subclasses MUST implement: fit, update, get_mass, get_edges.
    Optional capabilities (rebin_to, to_dense, to_sparse, to_vector, compare,
    sample, plot_*) are provided by mixing in the classes from hypergrid.mixin.
    BaseTensorHypergrid includes all mixins and is the recommended base.
    """

    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def fit(self, data, weights=None):
        """Build histogram from scratch."""

    @abstractmethod
    def update(self, data, weights=None):
        """Accumulate new data into the existing histogram."""

    @abstractmethod
    def get_mass(self):
        """Return the histogram as {tuple_index: float}."""

    @abstractmethod
    def get_edges(self):
        """Return list of edge arrays, one per dimension."""
