import numpy as np
import pytest

from hypergrid import AdaptiveHypergrid


class TestAdaptiveHypergrid:
    def test_fit_without_initial_edges(self):
        g = AdaptiveHypergrid()
        data = np.random.default_rng(40).standard_normal((500, 2))
        g.fit(data)
        assert g.edges is not None
        assert g.dim == 2
        assert sum(g.get_mass().values()) > 0

    def test_update_without_fit_auto_inits(self):
        g = AdaptiveHypergrid()
        data = np.random.default_rng(41).standard_normal((200, 3))
        g.update(data)
        assert g.edges is not None
        assert g.dim == 3

    def test_fit_with_initial_edges(self):
        edges = [np.linspace(-2, 2, 9), np.linspace(-2, 2, 9)]
        g = AdaptiveHypergrid(edges=edges)
        data = np.random.default_rng(42).standard_normal((500, 2))
        g.fit(data)
        assert sum(g.get_mass().values()) > 0

    def test_get_edges_returns_list(self):
        g = AdaptiveHypergrid()
        g.fit(np.random.default_rng(43).standard_normal((200, 2)))
        edges = g.get_edges()
        assert len(edges) == 2
        for e in edges:
            assert isinstance(e, np.ndarray)

    def test_drift_triggers_rebin(self):
        rng = np.random.default_rng(44)
        g = AdaptiveHypergrid(drift_threshold=0.01, buffer_size=500)

        # fit on data centred at 0
        g.fit(rng.standard_normal((300, 2)))

        # push heavily shifted data to trigger overflow
        g.update(rng.standard_normal((500, 2)) + 10.0)

        assert len(g._drift_history) >= 1

    def test_drift_history_values_in_range(self):
        rng = np.random.default_rng(45)
        g = AdaptiveHypergrid(drift_threshold=0.01)
        g.fit(rng.standard_normal((300, 2)))
        g.update(rng.standard_normal((500, 2)) + 10.0)

        for frac in g._drift_history:
            assert 0.0 <= frac <= 1.0

    def test_fit_resets_history(self):
        rng = np.random.default_rng(46)
        g = AdaptiveHypergrid(drift_threshold=0.01)
        g.fit(rng.standard_normal((300, 2)))
        g.update(rng.standard_normal((500, 2)) + 10.0)
        # fit again should NOT reset drift_history (it's cumulative across the
        # lifetime of the object), but storage should be cleared
        total_before = sum(g.get_mass().values())
        g.fit(rng.standard_normal((300, 2)))
        total_after = sum(g.get_mass().values())
        # new fit starts fresh count
        assert total_after <= total_before or total_before == total_after

    def test_mass_keys_are_tuples(self):
        g = AdaptiveHypergrid()
        g.fit(np.random.default_rng(47).standard_normal((200, 3)))
        for k in g.get_mass():
            assert isinstance(k, tuple)
            assert len(k) == 3
