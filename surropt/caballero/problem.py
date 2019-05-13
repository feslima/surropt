from pydace import predictor
import numpy as np


class CaballeroProblem(object):
    """Class problem interface for IpOpt use."""
    def __init__(self, obj_surrogate: dict, con_surrogate: list):
        self._obj_surr = obj_surrogate
        self._con_surr = con_surrogate

    def objective(self, x):
        f, *_ = predictor(x, self._obj_surr)

        return f

    def gradient(self, x):
        _, gf, *_ = predictor(x, self._obj_surr, compute_jacobian=True)

        return gf

    def constraints(self, x):
        c = np.zeros((len(self._con_surr), 1))

        for i in range(len(self._con_surr)):
            c[i, 0], *_ = predictor(x, self._con_surr[i])

        return c

    def jacobian(self, x):
        gc = np.zeros((len(self._con_surr), x.size))

        for i in range(len(self._con_surr)):
            _, gc_col, *_ = predictor(x, self._con_surr[i], compute_jacobian=True)
            gc[[i], :] = gc_col.reshape(1, -1)

        return gc


def objective_prediction(x, surrmodel):
    """Function objective prediction for optimization. Only accepts pydace surrogate models.
    Parameters
    ----------
    x : ndarray
        Input variable vector.
    surrmodel : dict
        pydace surrogate model of objective function.

    Returns
    -------
    f : float
        Objective function evaluation for given `x`.
    g : ndarray
        Gradient column vector evaluation for given `x` (as 1D array).
    """
    f, g, *_ = predictor(x, surrmodel, compute_jacobian='yes')

    return f, g.flatten()


def constraint_prediction(x, surrmodel: list):
    """Constraint function for optimization. Only accepts pydace surrogate models.
    Parameters
    ----------
    x : ndarray
        Input variable vector.
    surrmodel : list
        pydace surrogate model of constraints.

    Returns
    -------
    f : float
        Objective function evaluation for given `x`.
    g : ndarray
        Gradient column vector evaluation for given `x`.
    """
    c = np.zeros((len(surrmodel), 1))  # has to be a column array
    gc = np.zeros((len(surrmodel), x.size))  # m-by-n array (n is number of variables and m is number of constraints)
    for i in np.arange(len(surrmodel)):
        ph = predictor(x, surrmodel[i], compute_jacobian='yes')[:2]
        c[i, 0] = ph[0]
        gc[[i], :] = ph[1].reshape(-1)

    return c, np.array([[]]), gc, np.array([[]])
