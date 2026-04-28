import numpy as np
import pytest

from hypergrid import DenseHypergrid, SparseHypergrid
from hypergrid.utils.binning import compute_edges


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def data_3d(rng):
    return rng.standard_normal((2000, 3))


@pytest.fixture(scope="session")
def edges_3d(data_3d):
    return compute_edges(data_3d)


@pytest.fixture
def dense_grid(edges_3d, data_3d):
    g = DenseHypergrid(edges_3d)
    g.fit(data_3d)
    return g


@pytest.fixture
def sparse_grid(edges_3d, data_3d):
    g = SparseHypergrid(edges_3d)
    g.fit(data_3d)
    return g


@pytest.fixture
def simple_edges():
    return [np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0])]


@pytest.fixture
def simple_data():
    return np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 1.5], [0.5, 1.5]])
