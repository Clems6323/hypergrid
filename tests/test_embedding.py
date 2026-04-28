import numpy as np
import pytest

from hypergrid import DenseHypergrid, SparseHypergrid


@pytest.fixture
def filled_grid():
    edges = [np.linspace(-2, 2, 9), np.linspace(-2, 2, 9)]
    data = np.random.default_rng(30).standard_normal((500, 2))
    g = DenseHypergrid(edges)
    g.fit(data)
    return g, edges


class TestToVector:
    def test_shape(self, filled_grid):
        g, edges = filled_grid
        n_bins = (len(edges[0]) - 1) * (len(edges[1]) - 1)
        vec = g.to_vector()
        assert vec.shape == (n_bins,)

    def test_sums_to_one(self, filled_grid):
        g, _ = filled_grid
        vec = g.to_vector()
        assert vec.sum() == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self, filled_grid):
        g, _ = filled_grid
        assert np.all(g.to_vector() >= 0)

    def test_same_edges_comparable(self):
        edges = [np.linspace(-2, 2, 5), np.linspace(-2, 2, 5)]
        rng = np.random.default_rng(31)
        g1 = DenseHypergrid(edges)
        g2 = SparseHypergrid(edges)
        data = rng.standard_normal((300, 2))
        g1.fit(data)
        g2.fit(data)
        np.testing.assert_allclose(g1.to_vector(), g2.to_vector(), atol=1e-12)

    def test_different_distributions_differ(self):
        edges = [np.linspace(-3, 3, 13), np.linspace(-3, 3, 13)]
        rng = np.random.default_rng(32)
        g1 = DenseHypergrid(edges)
        g2 = DenseHypergrid(edges)
        g1.fit(rng.standard_normal((500, 2)))
        g2.fit(rng.standard_normal((500, 2)) + 2.0)
        assert not np.allclose(g1.to_vector(), g2.to_vector())

    def test_empty_grid_returns_zero_vector(self):
        edges = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]
        g = DenseHypergrid(edges)
        g.fit(np.full((10, 2), 999.0))  # all out-of-bounds
        vec = g.to_vector()
        assert vec.sum() == pytest.approx(0.0, abs=1e-10)
