import numpy as np
import pytest

from hypergrid import DenseHypergrid, SparseHypergrid, StaticHypergrid
from hypergrid.utils.binning import compute_edges


GRID_CLASSES = [DenseHypergrid, SparseHypergrid, StaticHypergrid]


@pytest.fixture(params=GRID_CLASSES, ids=["dense", "sparse", "static"])
def grid_class(request):
    return request.param


@pytest.fixture
def edges_2d():
    return [np.linspace(-3, 3, 13), np.linspace(-3, 3, 13)]


@pytest.fixture
def data_2d():
    return np.random.default_rng(99).standard_normal((500, 2))


class TestFitUpdate:
    def test_fit_populates_mass(self, grid_class, edges_2d, data_2d):
        g = grid_class(edges_2d)
        g.fit(data_2d)
        mass = g.get_mass()
        assert len(mass) > 0
        assert sum(mass.values()) > 0

    def test_fit_resets_previous(self, grid_class, edges_2d, data_2d):
        g = grid_class(edges_2d)
        g.fit(data_2d)
        total_before = sum(g.get_mass().values())
        g.fit(data_2d)
        total_after = sum(g.get_mass().values())
        assert total_before == pytest.approx(total_after)

    def test_update_accumulates(self, grid_class, edges_2d, data_2d):
        g = grid_class(edges_2d)
        half = len(data_2d) // 2
        g.fit(data_2d[:half])
        total_half = sum(g.get_mass().values())
        g.update(data_2d[half:])
        total_full = sum(g.get_mass().values())
        assert total_full > total_half

    def test_out_of_bounds_ignored(self, grid_class, edges_2d):
        g = grid_class(edges_2d)
        oob = np.array([[100.0, 100.0], [200.0, 200.0]])
        g.fit(oob)
        assert sum(g.get_mass().values()) == 0.0

    def test_weighted_fit(self, grid_class, edges_2d):
        g = grid_class(edges_2d)
        data = np.array([[0.0, 0.0], [0.0, 0.0]])
        weights = np.array([3.0, 7.0])
        g.fit(data, weights=weights)
        total = sum(g.get_mass().values())
        assert total == pytest.approx(10.0)

    def test_single_point_1d_row(self, grid_class, edges_2d):
        g = grid_class(edges_2d)
        g.fit(np.array([0.5, 0.5]))
        assert sum(g.get_mass().values()) == pytest.approx(1.0)

    def test_get_edges_returns_list(self, grid_class, edges_2d, data_2d):
        g = grid_class(edges_2d)
        g.fit(data_2d)
        edges = g.get_edges()
        assert len(edges) == 2
        for e in edges:
            assert isinstance(e, np.ndarray)

    def test_dim_attribute(self, grid_class, edges_2d):
        g = grid_class(edges_2d)
        assert g.dim == 2

    def test_mass_keys_are_tuples(self, grid_class, edges_2d, data_2d):
        g = grid_class(edges_2d)
        g.fit(data_2d)
        for k in g.get_mass():
            assert isinstance(k, tuple)
            assert len(k) == 2
