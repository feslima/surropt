import numpy as np
import pytest

from surropt.caballero.problem import is_inside_hypercube

# tuple follows: (point, lower bound, upper bound, correct result)
_array_list = [([9.429, 24.721, 400, 217.738],
                [8.5, 0.0, 102.0, 0.0],
                [20.0, 100.0, 400.0, 400.0], False),
               ([-3, -3], [-5, -4], [-2, -1], True),
               ([-5.1, -3], [-5, -4], [-2, -1], 'Error'),
               ([5e-7, 1.1e-6], [0, 0], [1e-6, 1e-6], 'Error'),
               ([5e-7, 1e-6], [0, 0], [1e-6, 1e-6], False),
               ([5e-7, 5e-7], [0, 0], [1e-6, 1e-6], True)]


@pytest.mark.parametrize("p,lb,ub, correct_result", _array_list)
def test_is_inside(p, lb, ub, correct_result):
    p = np.array(p)
    lb = np.array(lb)
    ub = np.array(ub)

    if type(correct_result) is bool:
        assert is_inside_hypercube(p, lb, ub) == correct_result
    else:
        with pytest.raises(ValueError):
            is_inside_hypercube(p, lb, ub)
