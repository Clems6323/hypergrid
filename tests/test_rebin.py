import numpy as np
import pytest

from hypergrid import DenseHypergrid


@pytest.fixture
def source_grid():
    edges = [np.linspace(-2, 2, 9), np.linspace(-2, 2, 9)]
    data = np.random.default_rng(10).standard_normal((1000, 2))
    g = DenseHypergrid(edges)
    g.fit(data)
    return g


class TestRebinTo:
    def test_returns_dict(self, source_grid):
        target = [np.linspace(-2, 2, 5), np.linspace(-2, 2, 5)]
        result = source_grid.rebin_to(target)
        assert isinstance(result, dict)

    def test_keys_are_tuples(self, source_grid):
        target = [np.linspace(-2, 2, 5), np.linspace(-2, 2, 5)]
        result = source_grid.rebin_to(target)
        for k in result:
            assert isinstance(k, tuple)
            assert len(k) == 2

    def test_mass_conserved(self, source_grid):
        original_mass = sum(source_grid.get_mass().values())
        target = [np.linspace(-2, 2, 5), np.linspace(-2, 2, 5)]
        result = source_grid.rebin_to(target)
        rebinned_mass = sum(result.values())
        assert original_mass == pytest.approx(rebinned_mass, rel=1e-6)

    def test_coarser_grid_fewer_bins(self, source_grid):
        target = [np.linspace(-2, 2, 3), np.linspace(-2, 2, 3)]
        result = source_grid.rebin_to(target)
        assert len(result) <= 4  # at most 2×2 = 4 non-empty bins

    def test_identical_edges_same_mass(self, source_grid):
        target = source_grid.get_edges()
        result = source_grid.rebin_to(target)
        original = source_grid.get_mass()
        assert sum(result.values()) == pytest.approx(sum(original.values()), rel=1e-6)

    def test_as_storage_filled(self, source_grid):
        from hypergrid.storage.storage import DictStorage
        target = [np.linspace(-2, 2, 5), np.linspace(-2, 2, 5)]
        storage = DictStorage()
        source_grid.rebin_to(target, as_storage=storage)
        assert sum(storage.values()) > 0
