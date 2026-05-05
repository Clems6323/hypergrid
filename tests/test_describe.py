import numpy as np
import pytest

pytest.importorskip("pandas")
import pandas as pd

from hypergrid import DenseHypergrid, SparseHypergrid, AdaptiveHypergrid, TemporalHypergrid
from hypergrid.utils.binning import compute_edges


@pytest.fixture(scope="module")
def filled_grid():
    rng = np.random.default_rng(100)
    data = rng.standard_normal((3000, 3))
    edges = compute_edges(data)
    g = DenseHypergrid(edges)
    g.fit(data)
    return g, data


class TestDescribeBasic:
    def test_returns_dataframe(self, filled_grid):
        g, _ = filled_grid
        result = g.describe()
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_dimension_indices(self, filled_grid):
        g, _ = filled_grid
        assert list(g.describe().columns) == [0, 1, 2]

    def test_default_rows(self, filled_grid):
        g, _ = filled_grid
        df = g.describe()
        assert list(df.index) == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    def test_count_equals_total_mass(self, filled_grid):
        g, _ = filled_grid
        total = sum(g.get_mass().values())
        df = g.describe()
        for col in df.columns:
            assert df.loc["count", col] == pytest.approx(total)

    def test_mean_within_grid_range(self, filled_grid):
        g, _ = filled_grid
        df = g.describe()
        edges = g.get_edges()
        for d in df.columns:
            e = edges[d]
            assert e[0] <= df.loc["mean", d] <= e[-1]

    def test_std_non_negative(self, filled_grid):
        g, _ = filled_grid
        df = g.describe()
        for d in df.columns:
            assert df.loc["std", d] >= 0.0

    def test_min_le_median_le_max(self, filled_grid):
        g, _ = filled_grid
        df = g.describe()
        for d in df.columns:
            assert df.loc["min", d] <= df.loc["50%", d] <= df.loc["max", d]

    def test_percentile_ordering(self, filled_grid):
        g, _ = filled_grid
        df = g.describe()
        for d in df.columns:
            assert df.loc["25%", d] <= df.loc["50%", d] <= df.loc["75%", d]

    def test_percentiles_within_min_max(self, filled_grid):
        g, _ = filled_grid
        df = g.describe()
        for d in df.columns:
            for pct in ["25%", "50%", "75%"]:
                assert df.loc["min", d] <= df.loc[pct, d] <= df.loc["max", d]


class TestDescribeCustomPercentiles:
    def test_custom_percentiles_in_index(self, filled_grid):
        g, _ = filled_grid
        df = g.describe(percentiles=[0.1, 0.9])
        assert "10%" in df.index
        assert "90%" in df.index
        assert "50%" not in df.index

    def test_empty_percentiles(self, filled_grid):
        g, _ = filled_grid
        df = g.describe(percentiles=[])
        assert list(df.index) == ["count", "mean", "std", "min", "max"]

    def test_single_percentile(self, filled_grid):
        g, _ = filled_grid
        df = g.describe(percentiles=[0.5])
        assert "50%" in df.index

    def test_percentile_label_formatting(self, filled_grid):
        g, _ = filled_grid
        df = g.describe(percentiles=[0.333])
        assert "33.3%" in df.index

    def test_unsorted_percentiles_sorted_in_output(self, filled_grid):
        g, _ = filled_grid
        df = g.describe(percentiles=[0.75, 0.25, 0.5])
        pct_rows = [r for r in df.index if r.endswith("%")]
        values = {d: [df.loc[r, d] for r in pct_rows] for d in df.columns}
        for d, vals in values.items():
            assert vals == sorted(vals)

    def test_mean_close_to_data_mean(self):
        # A very fine grid should give mean close to the true data mean.
        rng = np.random.default_rng(101)
        data = rng.standard_normal((5000, 2))
        edges = [np.linspace(-4, 4, 81), np.linspace(-4, 4, 81)]
        g = DenseHypergrid(edges)
        g.fit(data)
        df = g.describe()
        for d in range(2):
            assert df.loc["mean", d] == pytest.approx(data[:, d].mean(), abs=0.15)


class TestDescribeBackends:
    @pytest.mark.parametrize("cls", [DenseHypergrid, SparseHypergrid])
    def test_same_result_across_backends(self, cls):
        rng = np.random.default_rng(102)
        data = rng.standard_normal((1000, 2))
        edges = compute_edges(data)
        g = cls(edges)
        g.fit(data)
        df = g.describe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 2

    def test_adaptive_hypergrid(self):
        rng = np.random.default_rng(103)
        g = AdaptiveHypergrid()
        g.fit(rng.standard_normal((500, 2)))
        df = g.describe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [0, 1]

    def test_temporal_hypergrid_delegates(self):
        rng = np.random.default_rng(104)
        edges = [np.linspace(-3, 3, 13)] * 2
        tg = TemporalHypergrid(DenseHypergrid(edges))
        tg.update(rng.standard_normal((500, 2)))
        df = tg.describe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [0, 1]


class TestDescribeEdgeCases:
    def test_empty_grid_returns_nan(self):
        edges = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]
        g = DenseHypergrid(edges)
        g.fit(np.full((10, 2), 999.0))  # all OOB
        df = g.describe()
        assert df.loc["count", 0] == 0.0
        assert pd.isna(df.loc["mean", 0])
        assert pd.isna(df.loc["std", 0])

    def test_single_bin_occupied(self):
        edges = [np.linspace(0, 3, 4), np.linspace(0, 3, 4)]
        g = DenseHypergrid(edges)
        g.fit(np.array([[0.5, 0.5]] * 10))
        df = g.describe()
        assert df.loc["count", 0] == pytest.approx(10.0)
        assert df.loc["std", 0] == pytest.approx(0.0)

    def test_weighted_count(self):
        edges = [np.linspace(0, 3, 4), np.linspace(0, 3, 4)]
        g = DenseHypergrid(edges)
        data = np.array([[0.5, 0.5], [1.5, 1.5]])
        g.fit(data, weights=np.array([3.0, 7.0]))
        df = g.describe()
        assert df.loc["count", 0] == pytest.approx(10.0)

    def test_1d_grid(self):
        x = np.random.default_rng(105).standard_normal((500, 1))
        edges = compute_edges(x)
        g = DenseHypergrid(edges)
        g.fit(x)
        df = g.describe()
        assert list(df.columns) == [0]
        assert df.loc["count", 0] > 0
