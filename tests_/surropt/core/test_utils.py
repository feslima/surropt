import numpy as np
import pytest
from surropt.core.utils import is_row_member

_b = np.array([[1, 4.], [3, 6], [5, 9], [2, 8]])
_a1 = np.array([1., 3])
_a2 = np.array([5., 9])
_a3 = np.array([8., 2])
_a4 = np.array([3., 6])
_a5 = np.vstack((_a1, _a4))
_a6 = np.vstack((_a3, _a1))
_a7 = np.vstack((_a2, _a3))
_a8 = np.vstack((_a2, _a2))
_a9 = np.vstack((_a4, _a2))
_is_row_member_array_list = [(_a1, _b, False),
                             (_a2, _b, True),
                             (_a3, _b, False),
                             (_a4, _b, True),
                             (_a5, _b, np.array([False, True])),
                             (_a6, _b, np.array([False, False])),
                             (_a7, _b, np.array([True, False])),
                             (_a8, _b, np.array([True, True])),
                             (_a9, _b, np.array([True, True]))]


@pytest.mark.parametrize("a,b,correct_result", _is_row_member_array_list)
def test_is_row_member(a, b, correct_result):
    if a.ndim == 1:
        assert is_row_member(a, b) == correct_result
    else:
        assert np.array_equal(is_row_member(a, b), correct_result)
