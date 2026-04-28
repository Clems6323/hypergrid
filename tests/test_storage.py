import numpy as np
import pytest

from hypergrid.storage.storage import DictStorage, DenseTensorStorage, SparseTensorStorage


class TestDictStorage:
    def test_add_and_get(self):
        s = DictStorage()
        s.add((0, 1), 3.0)
        s.add((0, 1), 2.0)
        assert s.get((0, 1)) == 5.0

    def test_get_missing_returns_default(self):
        s = DictStorage()
        assert s.get((9, 9), default=0.0) == 0.0

    def test_scale(self):
        s = DictStorage()
        s.add((0,), 10.0)
        s.scale(0.5)
        assert s.get((0,)) == 5.0

    def test_clear(self):
        s = DictStorage()
        s.add((0,), 1.0)
        s.clear()
        assert s.get((0,)) == 0.0
        assert list(s.items()) == []

    def test_to_dict(self):
        s = DictStorage()
        s.add((1, 2), 7.0)
        d = s.to_dict()
        assert isinstance(d, dict)
        assert d[(1, 2)] == 7.0

    def test_items_values(self):
        s = DictStorage()
        s.add((0,), 4.0)
        s.add((1,), 6.0)
        assert dict(s.items()) == {(0,): 4.0, (1,): 6.0}
        assert sorted(s.values()) == [4.0, 6.0]


class TestDenseTensorStorage:
    def test_add_and_get(self):
        s = DenseTensorStorage([3, 3])
        s.add((1, 2), 5.0)
        assert s.get((1, 2)) == 5.0

    def test_get_out_of_bounds_returns_default(self):
        s = DenseTensorStorage([2, 2])
        assert s.get((5, 5), default=-1.0) == -1.0

    def test_scale(self):
        s = DenseTensorStorage([2, 2])
        s.add((0, 0), 8.0)
        s.scale(0.25)
        assert s.get((0, 0)) == pytest.approx(2.0)

    def test_clear(self):
        s = DenseTensorStorage([2, 2])
        s.add((0, 0), 1.0)
        s.clear()
        assert s.get((0, 0)) == 0.0

    def test_to_dict_excludes_zeros(self):
        s = DenseTensorStorage([3, 3])
        s.add((0, 1), 2.0)
        d = s.to_dict()
        assert (0, 1) in d
        assert all(v != 0 for v in d.values())

    def test_items_nonzero_only(self):
        s = DenseTensorStorage([3, 3])
        s.add((2, 2), 1.0)
        keys = [k for k, _ in s.items()]
        assert (2, 2) in keys
        assert len(keys) == 1


class TestSparseTensorStorage:
    def test_add_and_get(self):
        s = SparseTensorStorage([4, 4])
        s.add((2, 3), 9.0)
        assert s.get((2, 3)) == 9.0

    def test_bounds_checking_raises(self):
        s = SparseTensorStorage([3, 3])
        with pytest.raises(IndexError):
            s.add((3, 0), 1.0)

    def test_get_missing_returns_default(self):
        s = SparseTensorStorage([4, 4])
        assert s.get((1, 1), 0.0) == 0.0

    def test_scale(self):
        s = SparseTensorStorage([3, 3])
        s.add((0, 0), 10.0)
        s.scale(0.1)
        assert s.get((0, 0)) == pytest.approx(1.0)

    def test_clear(self):
        s = SparseTensorStorage([3, 3])
        s.add((0, 0), 5.0)
        s.clear()
        assert s.get((0, 0)) == 0.0

    def test_to_dict(self):
        s = SparseTensorStorage([3, 3])
        s.add((1, 2), 3.0)
        d = s.to_dict()
        assert d[(1, 2)] == 3.0

    def test_to_dense(self):
        s = SparseTensorStorage([2, 2])
        s.add((0, 1), 5.0)
        arr = s.to_dense()
        assert arr.shape == (2, 2)
        assert arr[0, 1] == 5.0
