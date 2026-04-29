import numpy as np
import pytest

from hypergrid import DenseHypergrid, SparseHypergrid


@pytest.fixture
def edges():
    return [np.linspace(-3, 3, 13), np.linspace(-3, 3, 13)]


@pytest.fixture
def identical_grids(edges):
    data = np.random.default_rng(20).standard_normal((1000, 2))
    g1 = DenseHypergrid(edges)
    g2 = DenseHypergrid(edges)
    g1.fit(data)
    g2.fit(data)
    return g1, g2


@pytest.fixture
def different_grids(edges):
    rng = np.random.default_rng(21)
    g1 = DenseHypergrid(edges)
    g2 = DenseHypergrid(edges)
    g1.fit(rng.standard_normal((1000, 2)))
    g2.fit(rng.standard_normal((1000, 2)) + 2.0)  # shifted distribution
    return g1, g2


class TestCompare:
    @pytest.mark.parametrize("method", ["l1", "kl", "js"])
    def test_identical_grids_zero(self, identical_grids, method):
        g1, g2 = identical_grids
        assert g1.compare(g2, method=method) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("method", ["l1", "kl", "js"])
    def test_different_grids_positive(self, different_grids, method):
        g1, g2 = different_grids
        assert g1.compare(g2, method=method) > 0.0

    def test_js_bounded(self, different_grids):
        g1, g2 = different_grids
        val = g1.compare(g2, method="js")
        assert 0.0 <= val <= 1.0 + 1e-9

    def test_l1_symmetric(self, different_grids):
        g1, g2 = different_grids
        assert g1.compare(g2, method="l1") == pytest.approx(g2.compare(g1, method="l1"))

    def test_js_symmetric(self, different_grids):
        g1, g2 = different_grids
        assert g1.compare(g2, method="js") == pytest.approx(g2.compare(g1, method="js"))

    def test_dimension_mismatch_raises(self, edges):
        edges_1d = [np.linspace(-3, 3, 7)]
        g1 = DenseHypergrid(edges)
        g2 = DenseHypergrid(edges_1d)
        g1.fit(np.random.default_rng(22).standard_normal((100, 2)))
        g2.fit(np.random.default_rng(22).standard_normal((100, 1)))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            g1.compare(g2)

    def test_unknown_method_raises(self, identical_grids):
        g1, g2 = identical_grids
        with pytest.raises(ValueError, match="Unknown method"):
            g1.compare(g2, method="bogus")

    def test_wasserstein_non_negative(self, different_grids):
        g1, g2 = different_grids
        # Use small grids so LP is fast
        small_edges = [np.linspace(-3, 3, 5), np.linspace(-3, 3, 5)]
        d1 = DenseHypergrid(small_edges)
        d2 = DenseHypergrid(small_edges)
        rng = np.random.default_rng(23)
        d1.fit(rng.standard_normal((200, 2)))
        d2.fit(rng.standard_normal((200, 2)) + 1.0)
        assert d1.compare(d2, method="wasserstein") >= 0.0

    def test_align_self(self, different_grids):
        g1, g2 = different_grids
        val = g1.compare(g2, method="l1", align="self")
        assert val > 0.0

    def test_compare_cross_backend(self, edges):
        rng = np.random.default_rng(24)
        data = rng.standard_normal((500, 2))
        g_dense = DenseHypergrid(edges)
        g_sparse = SparseHypergrid(edges)
        g_dense.fit(data)
        g_sparse.fit(data)
        assert g_dense.compare(g_sparse, method="js") == pytest.approx(0.0, abs=1e-10)


class TestCompareMarginal:
    def test_rebin_true_uses_self_edges(self, edges):
        rng = np.random.default_rng(25)
        g1 = DenseHypergrid(edges)
        g2 = DenseHypergrid(edges)
        g1.fit(rng.standard_normal((500, 2)))
        g2.fit(rng.standard_normal((500, 2)) + 1.0)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        g1.compare_marginal(g2, dim=0, ax=ax, rebin=True)
        plt.close(fig)

    def test_rebin_false_uses_native_edges(self, edges):
        rng = np.random.default_rng(26)
        other_edges = [np.linspace(-4, 4, 17), np.linspace(-4, 4, 17)]
        g1 = DenseHypergrid(edges)
        g2 = DenseHypergrid(other_edges)
        g1.fit(rng.standard_normal((500, 2)))
        g2.fit(rng.standard_normal((500, 2)) + 1.0)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        g1.compare_marginal(g2, dim=0, ax=ax, rebin=False)
        plt.close(fig)
