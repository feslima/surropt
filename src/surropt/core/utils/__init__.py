import numpy as np
from scipy.spatial.distance import pdist


def _is_numeric_array_like(value):
    """Returns True if the input is a valid float array.
    """
    if isinstance(value, np.ndarray):
        return True if value.dtype == float else False

    else:
        # value is not array, try to convert it to numpy float array
        try:
            value = np.asarray(value, dtype=float)
        except ValueError:
            # conversion failed, input can't be converted to float array
            return False
        finally:
            # conversion successful
            return True


def is_row_member(a: np.ndarray, b: np.ndarray):
    """If `a` is a 1D array, checks if `a` is a row of `b`. Otherwise, checks
    if each row of `a` is row of b (the result is a 1D bool array).

    Parameters
    ----------
    a : np.ndarray
        1D or 2D array to be checked. If 1D, treats `a` as single row.
    b : np.ndarray
        2D array where `a` is checked against.

    Returns
    -------
    out : bool or 1D bool array
        If `a` is 1D, the result is a single boolean value, else a 1D bool
        array where each element corresponds to a row of `a`.
    """
    if b.ndim != 2:
        raise ValueError("b has to be a 2D array.")

    if a.ndim == 1:
        # a is a 1D array, treat the entire array as row
        return (a == b).all(axis=1).any()
    elif a.ndim == 2:
        # a is a 2D array, return a bool 1D array
        a_rows, _ = a.shape
        result = np.full((a_rows,), False, dtype=bool)

        for row in range(a_rows):
            result[row] = is_row_member(a[row], b)

        return result

    else:
        raise ValueError("Row check valid only for 1D or 2D arrays.")


def point_domain_distance(p1: np.ndarray, p2: np.ndarray, lb: np.ndarray,
                          ub: np.ndarray) -> float:
    """Computes the euclidian distance between two points (`p1`, `p2`) inside a
    box domain defined by a lower bound (`lb`) and an upper bound (`ub`). The
    result is the normalized distance between `p1` and `p2` (i.e. the
    percentage of the domain range which the distance corresponds).

    Parameters
    ----------
    p1 : np.ndarray
        Point 1 (1D array).

    p2 : np.ndarray
        Point 2 (1D array).

    lb : np.ndarray
        Domain lower bound (1D array).

    ub : np.ndarray
        Domain upper bound (2D array).

    Returns
    -------
    dist : float
        Normalized euclidian distance.
    """

    if p1.ndim != 1 or p2.ndim != 1 or lb.ndim != 1 or ub.ndim != 1:
        raise ValueError("All input points must be 1D arrays.")

    return (pdist(np.vstack((p1, p2)), metric='euclidean') /
            pdist(np.vstack((lb, ub)), metric='euclidean')).item()


def get_samples_index(x: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    """Get the indexes of `x` that are inside the hypercube bounded by `lb` and
    `ub`.

    Parameters
    ----------
    x : np.ndarray
        Sample set (2D array).
    lb : np.ndarray
        Lower bound of hypercube (1D array).
    ub : np.ndarray
        Upper bound of hypercube (1D array).

    Returns
    -------
    out : np.ndarray
        Indexes (row numbers) of `x` that are between `lb` and `ub`.
    """
    idx = np.nonzero(np.all(np.logical_and(np.less_equal(x, ub),
                                           np.greater_equal(x, lb)),
                            axis=1))[0]

    return idx


class CaballeroProblem(object):
    """Class problem interface for IpOpt use."""

    def __init__(self, obj_surrogate: dict, con_surrogate: list):
        self._obj_surr = obj_surrogate
        self._con_surr = con_surrogate

    def objective(self, x):
        f, *_ = self._obj_surr.predict(x)

        return f

    def gradient(self, x):
        _, gf, *_ = self._obj_surr.predict(x, compute_jacobian=True)

        return gf

    def constraints(self, x):
        c = np.zeros((len(self._con_surr), 1))

        for i in range(len(self._con_surr)):
            c[i, 0], *_ = self._con_surr[i].predict(x)

        return c

    def jacobian(self, x):
        gc = np.zeros((len(self._con_surr), x.size))

        for i in range(len(self._con_surr)):
            _, gc_col, *_ = self._con_surr[i].predict(x, compute_jacobian=True)
            gc[[i], :] = gc_col.reshape(1, -1)

        return gc
