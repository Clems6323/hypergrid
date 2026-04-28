import numpy as np
import pytest

from hypergrid.utils.binning import compute_bins_1d, compute_edges


class TestComputeBins1d:
    def test_fd_returns_edges(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        edges = compute_bins_1d(x, method="fd")
        assert edges[0] <= x.min()
        assert edges[-1] >= x.max()
        assert len(edges) >= 3  # at least 2 bins

    def test_sturges(self):
        x = np.random.default_rng(1).standard_normal(200)
        edges = compute_bins_1d(x, method="sturges")
        assert len(edges) >= 2

    def test_sqrt(self):
        x = np.random.default_rng(2).standard_normal(100)
        edges = compute_bins_1d(x, method="sqrt")
        assert len(edges) >= 2

    def test_max_bins_cap(self):
        x = np.random.default_rng(3).standard_normal(100000)
        edges = compute_bins_1d(x, method="fd", max_bins=10)
        assert len(edges) - 1 <= 10

    def test_constant_dimension(self):
        x = np.full(50, 3.14)
        edges = compute_bins_1d(x)
        assert len(edges) == 2
        assert edges[0] < 3.14 < edges[1]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown binning method"):
            compute_bins_1d(np.arange(10), method="bogus")

    def test_edges_are_sorted(self):
        x = np.random.default_rng(4).standard_normal(300)
        edges = compute_bins_1d(x)
        assert np.all(np.diff(edges) > 0)


class TestComputeEdges:
    def test_2d_returns_two_edge_arrays(self):
        data = np.random.default_rng(5).standard_normal((500, 2))
        edges = compute_edges(data)
        assert len(edges) == 2
        for e in edges:
            assert isinstance(e, np.ndarray)
            assert e.ndim == 1

    def test_3d(self):
        data = np.random.default_rng(6).standard_normal((1000, 3))
        edges = compute_edges(data)
        assert len(edges) == 3

    def test_1d_input_broadcast(self):
        x = np.random.default_rng(7).standard_normal(200)
        edges = compute_edges(x)
        assert len(edges) == 1

    def test_per_dimension_coverage(self):
        data = np.random.default_rng(8).standard_normal((500, 2))
        edges = compute_edges(data)
        for d in range(2):
            assert edges[d][0] <= data[:, d].min()
            assert edges[d][-1] >= data[:, d].max()
