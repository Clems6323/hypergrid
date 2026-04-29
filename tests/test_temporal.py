import numpy as np
import pytest

from hypergrid import DenseHypergrid, TemporalHypergrid


@pytest.fixture
def base_grid():
    edges = [np.linspace(-3, 3, 13), np.linspace(-3, 3, 13)]
    return DenseHypergrid(edges)


class TestTemporalHypergrid:
    def test_snapshots_saved_at_interval(self, base_grid):
        # Each update() call is one batch; the modulo check fires once per call
        # when _counter is a multiple of snapshot_interval after the batch.
        tg = TemporalHypergrid(base_grid, snapshot_interval=100)
        for _ in range(5):
            tg.update(np.random.default_rng(50).standard_normal((100, 2)))
        assert len(tg.snapshots) == 5

    def test_snapshot_is_dict(self, base_grid):
        tg = TemporalHypergrid(base_grid, snapshot_interval=100)
        tg.update(np.random.default_rng(51).standard_normal((200, 2)))
        for snap in tg.snapshots:
            assert isinstance(snap, dict)

    def test_evolution_returns_list(self, base_grid):
        tg = TemporalHypergrid(base_grid, snapshot_interval=100)
        tg.update(np.random.default_rng(52).standard_normal((500, 2)))
        result = tg.evolution(method="js")
        assert isinstance(result, list)
        assert len(result) == len(tg.snapshots) - 1

    def test_evolution_requires_two_snapshots(self, base_grid):
        tg = TemporalHypergrid(base_grid, snapshot_interval=1000)
        tg.update(np.random.default_rng(53).standard_normal((200, 2)))
        assert tg.evolution() == []

    def test_evolution_non_negative(self, base_grid):
        tg = TemporalHypergrid(base_grid, snapshot_interval=100)
        tg.update(np.random.default_rng(54).standard_normal((500, 2)))
        for d in tg.evolution(method="js"):
            assert d >= 0.0

    def test_decay_reduces_old_counts(self):
        edges = [np.linspace(-3, 3, 13), np.linspace(-3, 3, 13)]
        g = DenseHypergrid(edges)
        tg = TemporalHypergrid(g, decay=0.5, snapshot_interval=10000)
        data = np.random.default_rng(55).standard_normal((100, 2))
        tg.update(data)
        total_after_first = sum(tg.get_mass().values())

        # Second update applies 0.5 decay before adding new data.
        # The prior mass should have been halved.
        tg_nodecay = TemporalHypergrid(DenseHypergrid(edges), decay=None, snapshot_interval=10000)
        tg_nodecay.grid.fit(data)
        total_nodecay = sum(tg_nodecay.get_mass().values())

        assert total_after_first < total_nodecay * 1.5  # decay had effect

    def test_dim_and_edges_forwarded(self, base_grid):
        tg = TemporalHypergrid(base_grid)
        assert tg.dim == base_grid.dim
        assert tg.edges is base_grid.edges

    def test_fit_resets_counter_and_snapshots(self, base_grid):
        tg = TemporalHypergrid(base_grid, snapshot_interval=100)
        tg.update(np.random.default_rng(56).standard_normal((500, 2)))
        assert len(tg.snapshots) > 0
        tg.fit(np.random.default_rng(57).standard_normal((200, 2)))
        assert len(tg.snapshots) == 0
        assert tg._counter == 0

    @pytest.mark.parametrize("method", ["js", "kl", "l1"])
    def test_evolution_methods(self, base_grid, method):
        tg = TemporalHypergrid(base_grid, snapshot_interval=100)
        tg.update(np.random.default_rng(58).standard_normal((500, 2)))
        result = tg.evolution(method=method)
        assert len(result) == len(tg.snapshots) - 1

    def test_evolution_wasserstein(self):
        edges = [np.linspace(-3, 3, 7), np.linspace(-3, 3, 7)]  # small grid for LP speed
        tg = TemporalHypergrid(DenseHypergrid(edges), snapshot_interval=100)
        for _ in range(3):
            tg.update(np.random.default_rng(59).standard_normal((100, 2)))
        result = tg.evolution(method="wasserstein")
        assert len(result) == len(tg.snapshots) - 1
        assert all(v >= 0.0 for v in result)
