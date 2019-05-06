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
