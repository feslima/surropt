import numpy as np
from scipy.optimize import root

from ..core.utils import _is_numeric_array_like


def evaporator(f1: float, f3: float, p100: float, f200: float,
               x1: float = 5.0, t1: float = 40.0, t200: float = 25.0,
               x0: list = None) -> dict:
    """Model function of an evaporation process.

    Parameters
    ----------
    f1 : float
        Feed flow rate value (kg/min).
    f3 : float
        Circulating flow rate (kg/min).
    p100 : float
        Steam pressure (kPa).
    f200 : float
        Cooling water flow rate (kg/min).
    x1 : float, optional
        Feed composition (molar %), by default 5.0.
    t1 : float, optional
        Feed temperature (C), by default 40.0.
    t200 : float, optional
        Inlet temperature of the cooling water (C), by default 25.0.
    x0 : list (floats), optional
        Initial estimate to be used by the non-linear solver. Must have 12
        elements.

    Returns
    -------
    results : dict
        Dictionary containing the results
    """

    def sub_model(x, params):
        f1, f3, p100, f200 = params

        f2, f4, f5, x2, t2, t3, p2, f100, t100, q100, t201, q200 = x

        quoc = (t3 - t200) / (0.14 * f200 + 6.84)

        eq = [
            (f1 - f4 - f2) / 20,
            (f1 * x1 - f2 * x2) / 20,
            (f4 - f5) / 4,
            0.5616 * p2 + 0.3126 * x2 + 48.43 - t2,
            0.507 * p2 + 55 - t3,
            (q100 - 0.07 * f1 * (t2 - t1)) / 38.5 - f4,
            0.1538 * p100 + 90 - t100,
            0.16 * (f1 + f3) * (t100 - t2) - q100,
            q100 / 36.6 - f100,
            0.9576 * f200 * quoc - q200,
            t200 + 13.68 * quoc - t201,
            q200 / 38.5 - f5
        ]

        return eq

    # initial estimate
    if x0 is None:
        x0 = (1000 * np.ones(12,)).tolist()
    else:
        # check dimension and type
        if _is_numeric_array_like(x0):
            x0 = np.asarray(x0, dtype=float)
        else:
            raise ValueError("'x0' has to be a float array.")

    # extra args
    param = (f1, f3, p100, f200)

    # solver call
    res = root(sub_model, x0, args=(param,))

    # unpack the results
    f2, f4, f5, x2, t2, t3, p2, f100, t100, q100, t201, q200 = \
        res['x'].tolist()

    # calculate the objective function
    j = 600 * f100 + 0.6 * f200 + 1.009 * (f2 + f3) + 0.2 * f1 - 4800 * f2

    # constraints
    g1 = 35.5 - x2
    g2 = p2 - 80
    g3 = 40 - p2

    g = [g1, g2, g3]

    # status of the run (True or False)
    status = res['success']

    # extra outputs (maybe useful/optional)
    extras = dict(zip(["f2", "f4", "f5", "x2", "t2", "t3", "p2", "f100",
                       "t100", "q100", "t201", "q200"], res['x'].tolist()))

    # results dictionary (must contain status, obj, const)
    results = {
        'status': status,
        'f': j,
        'g': g,
        'extras': extras
    }

    return results
