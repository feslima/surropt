import numpy as np
import scipy.io as sio
from scipy.optimize import root
from sympy import Float
from sympy.core.symbol import symbols
from sympy.solvers.solveset import nonlinsolve
from tests_ import RESOURCES_PATH


def evaporator_doe(x: np.ndarray):
    """

    Parameters
    ----------
    x : (M, 4) or (M, 7) ndarray
        Input data for the evaporator model. If the number of columns is 4, then the disturbances values are the nominal
        values from [1]_. Otherwise, the first three columns are the disturbances followed by F1, F3, P100 and F200
        values. The same order is applied when `x` has 4 columns.

    Returns
    -------
    sol : dict
        Dictionary containing the solution array and the status flag:

        'solution' : (M, 21) ndarray
            Output info about the evaporator model for a given set of inputs. The indexation is as follows:
            0 - X1;
            1 - T1;
            2 - T200;
            3 - F1;
            4 - F3;
            5 - P100;
            6 - F200;
            7 - F2;
            8 - F4;
            9 - F5;
            10 - X2;
            11 - T2;
            12 - T3;
            13 - P2;
            14 - F100;
            15 - T100;
            16 - Q100;
            17 - T201;
            18 - Q200;
            19 - J;
            20 - success (boolean - True for success, false for failure);

        'status' : bool
            Status of whether or not the solution is converged

    References
    ----------
    .. [1] Kariwala, V., Cao, Y. and Janardhanan, S. (2008). Local Self-Optimizing Control with Average Loss
        Minimization. Industrial & Engineering Chemistry Research, 47(4), pp.1150-1158.
    """
    def evaporator_model(k):
        nonlocal f1_lhs, f3_lhs, p100_lhs, f200_lhs, x1_lhs, t1_lhs, t200_lhs
        f1 = f1_lhs
        f2 = k[0]
        f3 = f3_lhs
        f4 = k[1]
        f5 = k[2]
        x1 = x1_lhs
        x2 = k[3]
        t1 = t1_lhs
        t2 = k[4]
        t3 = k[5]
        p2 = k[6]
        f100 = k[7]
        t100 = k[8]
        p100 = p100_lhs
        q100 = k[9]
        f200 = f200_lhs
        t200 = t200_lhs
        t201 = k[10]
        q200 = k[11]

        out = []
        out.append((f1 - f4 - f2) / 20)
        out.append((f1 * x1 - f2 * x2) / 20)
        out.append((f4 - f5) / 4)
        out.append(0.5616 * p2 + 0.3126 * x2 + 48.43 - t2)
        out.append(0.507 * p2 + 55 - t3)
        out.append((q100 - 0.07 * f1 * (t2 - t1)) / 38.5 - f4)
        out.append(0.1538 * p100 + 90 - t100)
        out.append(0.16 * (f1 + f3) * (t100 - t2) - q100)
        out.append(q100 / 36.6 - f100)
        quoc = (t3 - t200) / (0.14 * f200 + 6.84)
        out.append(0.9576 * f200 * quoc - q200)
        out.append(t200 + 13.68 * quoc - t201)
        out.append(q200 / 38.5 - f5)

        return out

    # input treatment
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be numpy ndarray.")
    else:
        if x.ndim == 1:  # 1d flattened array
            x = x.reshape(1, x.size)

    m, n = x.shape  # m is number of cases, n is the dimension

    if n == 4:
        # MV only
        cases = x.copy()  # F1 F3 P100 F200
        no = np.array([[5., 40., 25.]])  # X1 T1 T200
    elif n == 7:
        # disturbances included
        cases = x
        no = x[:, :3]
    else:
        raise ValueError("x must have 4 or  columns.")

    # pre-allocate the result matrix
    doe_build = np.zeros((m, 21))
    for i in np.arange(m):
        if n == 7:
            x1_lhs = cases[i, 0]
            t1_lhs = cases[i, 1]
            t200_lhs = cases[i, 2]
            f1_lhs = cases[i, 3]
            f3_lhs = cases[i, 4]
            p100_lhs = cases[i, 5]
            f200_lhs = cases[i, 6]
        else:
            x1_lhs = no[0, 0]
            t1_lhs = no[0, 1]
            t200_lhs = no[0, 2]
            f1_lhs = cases[i, 0]
            f3_lhs = cases[i, 1]
            p100_lhs = cases[i, 2]
            f200_lhs = cases[i, 3]

        # res = fsolve(evaporator_model, 10*np.ones((1, 12)), full_output=True)
        res = root(evaporator_model, 1000*np.ones((1, 12)))

        # indexation order
        # X1 T1 T200 F1 F3 P100 F200 F2 F4 F5 X2 T2 T3 P2 F100 T100 Q100 T201 Q200 J exiflag
        if n == 7:
            doe_build[i, :7] = cases[i, :]
        else:
            doe_build[i, :3] = no[0, :]
            doe_build[i, 3:7] = cases[i, :]

        x_out = res["x"]
        doe_build[i, 7:-2] = x_out
        doe_build[i, -2] = 600 * x_out[7] + 0.6 * f200_lhs + 1.009 * (x_out[0] + f3_lhs) + 0.2 * f1_lhs - 4800 * x_out[0]
        doe_build[i, -1] = res["status"] if np.all(x_out >= 0) else False

    return {'solution': doe_build[:, :-1], 'status': doe_build[:, -1]}


def evap_sympy(f1_lhs, f3_lhs, p100_lhs, f200_lhs):
    """Sympy version of the evaporator_doe routine.

    Parameters
    ----------
    f1_lhs : float
    f3_lhs : float
    p100_lhs : float
    f200_lhs : float

    Returns
    -------
    out : ndarray
        List containing solutions, if it is empty (then the system of equations has no solution). The indexation is as
        follows:
        0 - F2;
        1 - F4;
        2 - F5;
        3 - X2;
        4 - T2;
        5 - T3;
        6 - P2;
        7 - F100;
        8 - T100;
        9 - Q100;
        10 - T201;
        11 - Q200;
        12 - J;

    """
    f1 = f1_lhs
    f3 = f3_lhs
    x1 = 5.0
    t1 = 40.0
    p100 = p100_lhs
    f200 = f200_lhs
    t200 = 25.0

    f2, f4, f5, x2, t2, t3, p2, f100, t100, q100, t201, q200 = symbols(
        'f2, f4, f5, x2, t2, t3, p2, f100, t100, q100, t201, q200', real=True)

    quoc = (t3 - t200) / (0.14 * f200 + 6.84)
    out = []
    out.append((f1 - f4 - f2) / 20.)
    out.append((f1 * x1 - f2 * x2) / 20.)
    out.append((f4 - f5) / 4)
    out.append(0.5616 * p2 + 0.3126 * x2 + 48.43 - t2)
    out.append(0.507 * p2 + 55. - t3)
    out.append((q100 - 0.07 * f1 * (t2 - t1)) / 38.5 - f4)
    out.append(0.1538 * p100 + 90 - t100)
    out.append(0.16 * (f1 + f3) * (t100 - t2) - q100)
    out.append(q100 / 36.6 - f100)

    out.append(0.9576 * f200 * quoc - q200)
    out.append(t200 + 13.68 * quoc - t201)
    out.append(q200 / 38.5 - f5)

    sol = nonlinsolve(out, [f2, f4, f5, x2, t2, t3, p2, f100, t100, q100, t201, q200])

    out = []
    for arg in sol.args:  # traverse the finite set solution and capture numbers through recursion
        sol_list = []
        __tra_sol(arg, sol_list)

        if np.all(np.array(sol_list) >= 0):  # return the first solution with all positive values
            # calculate the profit value
            J = 600. * sol_list[7] + 0.6 * f200 + 1.009 * (sol_list[0] + f3) + 0.2 * f1 - 4800. * sol_list[0]
            sol_list.append(J)
            out = sol_list
            break

    return np.array(out)  # if out is empty, then the system doesn't have solution


def __tra_sol(solution, sol_list):
    for arg in solution.args:
        if not isinstance(arg, Float):  # has elements
            __tra_sol(arg, sol_list)
        else:  # is a float, capture the number
            sol_list.append(float(arg))


if __name__ == "__main__":
    # load the initial sampling
    mat_contents = sio.loadmat(RESOURCES_PATH / 'evap53pts.mat')
    # mat_contents = sio.loadmat(RESOURCES_PATH / 'evap5k.mat')
    doe_build = mat_contents["doeBuild"]

    check = np.zeros((doe_build.shape[0], ))
    for i in range(doe_build.shape[0]):
        f1_lhs = doe_build[i, 3].item()
        f3_lhs = doe_build[i, 4].item()
        p100_lhs = doe_build[i, 5].item()
        f200_lhs = doe_build[i, 6].item()

        # sol = evap_sympy(f1_lhs, f3_lhs, p100_lhs, f200_lhs)
        sol2 = evaporator_doe(doe_build[[i], :7])
        check[i] = np.allclose(doe_build[i, 7:], sol2[[0], 7:-1])

    print("All close" if np.all(check) else "Some cases are not converged")
