import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp as cvxqp
from cvxopt.solvers import options
from quadprog import solve_qp

from surropt.utils.matrixdivide import mldivide

options['show_progress'] = False  # disable cvxopt output
options['abstol'] = 1e-8
options['feastol'] = 1e-8


# Linesearch auxiliary routine
def __linesearch(x, p, objfun, nonlcon, lambdav, obj, objgrad, globalls):
    eta = 0.25
    tau = 0.5

    # Choose mu to satisfy 18.36 with sigma = 1
    delta_bar = np.sqrt(np.spacing(1))

    if lambdav.size == 0:  # no consraint multipliers
        mu = 1 / delta_bar
    else:
        mu = 1 / (np.linalg.norm(lambdav, np.inf) + delta_bar)

    alpha = 1.0

    c = objgrad

    d, ce, *_ = nonlcon(x)

    # call the merit function
    phi_x_mu, obj, objgrad, globalls = __merit_fun(obj, objgrad, objfun, nonlcon, x, mu, globalls)

    D_phi_x_mu = c.conj().T @ p  # directional derivative

    # only the elements of d corresponding to violated constraints should be included
    idx = d > 0
    if idx.size != 0:
        t = - np.linalg.norm(np.vstack((ce, d[idx].reshape(-1, 1))), 1) / mu
    else:
        t = - np.linalg.norm(ce, 1) / mu

    if t.size != 0:
        D_phi_x_mu += t

    while True:  # while loop to find alpha that causes decrease in merit
        p1, obj, objgrad, globalls = __merit_fun(np.array([]), objgrad, objfun, nonlcon, x + alpha * p, mu, globalls)
        p2 = phi_x_mu + eta * alpha * D_phi_x_mu

        if p1 > p2:
            # Reset alpha = tau_alpha * alpha for some tau_alpha in the range [0, 1]
            tau_alpha = 0.9 * tau
            alpha *= tau_alpha

        else:
            break  # the alpha is enough to cause decrease

    x_new = x + alpha * p

    return x_new, alpha, obj, objgrad, globalls


# Merit function evaluation auxiliary routine
def __merit_fun(obj, objgrad, objfun, nonlcon, x, mu, globalls):
    ci, ce, *_ = nonlcon(x)

    idx = ci > 0  # indexes of negative (violated) constraints

    if idx.size != 0:
        con = np.vstack((ce, ci[idx].reshape(-1, 1)))
    else:
        con = ce

    if type(obj) is np.ndarray and obj.size == 0:
        obj, objgrad = objfun(x)  # evaluate function if obj value is empty
        globalls['nevals'] += 1

    merit = obj
    t = np.linalg.norm(con, 1) / mu

    if t.size != 0:  # if t is not empty
        merit += t

    return merit, obj, objgrad, globalls


# Transform bounds into constraints (gradients included)
def __bnd2cf(x, lbidx, ubidx, lb, ub, bnds_grad, nonlcon):
    cif_eval, cef, cigf_eval, cegf = nonlcon(x)

    if cef.size == 0:
        cef = cef.reshape(-1, 1)  # in case of empty array, reshape it to avert vstack problems

    if cegf.size == 0:
        cegf = cegf.reshape(-1, x.size)  # same reshape logic as cef

    if cif_eval.size != 0:
        cif = np.vstack((cif_eval, lb[lbidx].reshape(-1, 1) - x[lbidx].reshape(-1, 1),
                         x[ubidx].reshape(-1, 1) - ub[ubidx].reshape(-1, 1)))
    else:
        cif = np.vstack((lb[lbidx].reshape(-1, 1) - x[lbidx].reshape(-1, 1),
                         x[ubidx].reshape(-1, 1) - ub[ubidx].reshape(-1, 1)))

        if cif.size == 0:  # boundless problem and without constraints
            cif = cif.reshape(-1, 1)  # to avoid concatenation problem

    if cigf_eval.size != 0:
        cigf = np.vstack((cigf_eval, bnds_grad))
    else:
        cigf = bnds_grad

    return cif, cef, cigf, cegf


# Check inputs auxiliaries (sanitize input vectors by enforcing them to be 1D vectors of same size)
def __check_vector_input(x0, lb, ub):
    # if any of the bounds is set to None, make them a infinity vector
    if lb is None:
        lb = - np.ones((x0.size, )) * np.inf
    else:
        lb = lb.flatten()

    if ub is None:
        ub = np.ones((x0.size, )) * np.inf
    else:
        ub = ub.flatten()

    x0 = x0.flatten()  # force to be 1D array

    if x0.size != lb.size or x0.size != ub.size:
        ValueError("x0, lb and ub has to have the same number of elements.")

    if np.any(lb >= ub):
        ValueError("All lower bounds must be lower than the upper bounds (lb < ub).")

    return x0, lb, ub


def __set_options_structure(options, x0):
    if options is None:
        # set the default values of the options structure
        maxiter = 400
        maxfunevals = 100 * x0.size
        tolstep = np.sqrt(np.spacing(1))
        tolopt = np.sqrt(np.spacing(1))
        tolcon = np.sqrt(np.spacing(1))
        qpsolver = 'quadprog'

        return maxiter, maxfunevals, tolstep, tolopt, tolcon, qpsolver

    elif type(options) is dict:  # user wants to set non-default values
        # check for the keys inside the dictionary
        valid_opts = ['maxiter', 'maxfunevals', 'tolstep', 'tolopt', 'tolcon']

        # get all keys set in dictionary that are invalid
        invalid_opts = [opt for opt in valid_opts if opt not in options]
        if len(invalid_opts) != 0:  # at least one invalid option was set. Throw exception
            ValueError("Invalid options were set. They are: {}".format(invalid_opts))

        # maximum number of iterations
        if 'maxiter' in options:
            maxiter = options['maxiter']
            if not np.isscalar(maxiter) or maxiter <= 0 or np.fix(maxiter) != maxiter:
                ValueError('Maximum number of iterations must be a positive integer scalar value.')
        else:
            maxiter = 400  # set to default

        # maximum number of function evaluations
        if 'maxfunevals' in options:
            maxfunevals = options['maxfunevals']
            if not np.isscalar(maxfunevals) or maxfunevals <= 0 or np.fix(maxfunevals) != maxfunevals:
                ValueError('Maximum number of function evaluations must be a positive integer scalar value.')

        else:
            maxfunevals = 100 * x0.size

        # minimum step size
        if 'tolstep' in options:
            tolstep = options['tolstep']
            if not np.isscalar(tolstep) or tolstep <= 0:
                ValueError('Invalid value set for step size tolerance.')

        else:
            tolstep = 1e-6

        # optimality tolerance
        if 'tolopt' in options:
            tolopt = options['tolopt']
            if not np.isscalar(tolopt) or tolopt <= 0:
                ValueError('Invalid value set for optimality tolerance.')

        else:
            tolopt = 1e-6

        # constraint tolerance
        if 'tolcon' in options:
            tolcon = options['tolcon']
            if not np.isscalar(tolcon) or tolcon <= 0:
                ValueError('Invalid value set for constraint tolerance.')

        else:
            tolcon = 1e-6

        # QP solver
        if 'qpsolver' in options:
            if isinstance(options['qpsolver'], str):  # if it is a string, check for valid options
                qpsolver = options['qpsolver']
                if qpsolver != 'cvxopt' or qpsolver != 'quadprog':
                    ValueError("Invalid QP solver option.")

        else:
            qpsolver = 'quadprog'

        return maxiter, maxfunevals, tolstep, tolopt, tolcon, qpsolver

    else:  # invalid argument
        ValueError("options argument has to be a dictionary.")


def __objective_function_check(objfun, x0: np.ndarray):
    if callable(objfun):
        obj, gradobj = objfun(x0)

        if not isinstance(obj, np.ndarray):
            obj = np.asarray(obj)

        if not isinstance(gradobj, np.ndarray):
            gradobj = np.asarray(gradobj)

        if obj.size != 1:
            ValueError("The objective function evaluation must return a single value (scalar)")

        if gradobj.ndim > 1 or gradobj.size == 0 or gradobj.size != x0.size:
            ValueError("The gradient evaluation has to return a (non-empty) 1D array of the same size as x0.")
    else:
        TypeError("Objective function and its gradient must be a callable function")


def __constraint_function_check(confun, x0):
    if callable(confun):  # if confun is a callable function
        cif_eval, cef_eval, cigf_eval, cegf_eval = confun(x0)

        if not isinstance(cif_eval, np.ndarray):
            cif_eval = np.asarray(cif_eval)  # convert to numpy array

        if not isinstance(cef_eval, np.ndarray):
            cef_eval = np.asarray(cef_eval)

        if not isinstance(cigf_eval, np.ndarray):
            cigf_eval = np.asarray(cigf_eval)

        if not isinstance(cegf_eval, np.ndarray):
            cegf_eval = np.asarray(cegf_eval)

        # check if constraint functions are returning column vectors
        if cif_eval.ndim != 2:  # 2D
            ValueError("Inequality constraint function must return a 2D array. The current function is returning "
                       f"{cif_eval.ndim} dimension(s)")
        else:
            if cif_eval.shape[1] == 1 or cif_eval.shape[1] == 0:
                ValueError("Inequality constraint function must return a column array.")

        if cef_eval.ndim != 2:  # 2D
            ValueError("Equality constraint function must return a 2D array. The current function is returning "
                       f"{cef_eval.ndim} dimension(s)")
        else:
            if cef_eval.shape[1] == 1 or cef_eval.shape[1] == 0:
                ValueError("Inequality constraint function must return a column array.")

        # check if the jacobian is specified without its corresponding function evaluation
        if (cif_eval.size == 0 and cigf_eval.size != 0) or (cef_eval.size == 0 and cegf_eval.size != 0):
            ValueError("You can't specify the jacobian without their constraint functions")

        # check if the jacobians have their proper dimensions
        n_ce, n_var = cegf_eval.shape
        if n_ce != 0 and (n_ce != cef_eval.size or n_var != x0.size):
            ValueError("The jacobian of equality constraints must have its number of rows equal to the number "
                       "of equality constraint functions, and its number of columns equal to the number of elements "
                       "of the initial estimate.")

        n_ci, n_var = cigf_eval.shape
        if n_ci != 0 and (n_ci != cif_eval.size or n_var != x0.size):
            ValueError("The jacobian of inequality constraints must have its number of rows equal to the number "
                       "of inequality constraint functions, and its number of columns equal to the number of elements "
                       "of the initial estimate.")

    else:
        ValueError("Constraints function and their jacobian must be a callable function.")


# Empty non-linear constraint function (if nonlcon is empty)
def __empty_nonlcon():
    empty_eval = np.array([[]])
    return empty_eval, empty_eval, empty_eval, empty_eval


def __qp_solve(H, f, A=None, b=None, Aeq=None, beq=None, x0=None, solver=None):
    """
    Finds a minimum for a problem specified by
    min 0.5 * x^T * H * x + f^T * x
    subject to:
    A*x <= b
    Aeq*x = beq

    Parameters
    ----------
    H : numpy.array
        Symmetric array of doubles.
    f : numpy.array
        One-dimensional array of doubles.
    A : numpy.array
        Two-dimensional array of doubles. Linear coefficients of the inequality constraints.
    b : numpy.array
        One-dimensional array of doubles. Constant vector of the inequality constraints.
    Aeq : numpy.array
        Two-dimensional array of doubles. Linear coefficients of the equality constraints.
    beq : numpy.array
        One-dimensional array of doubles. Constant vector of the equality constraints.
    x0 : numpy.array
        One-dimensional array of doubles. Warm-start guess vector.
    solver: string
        Set to 'cvxopt' to run CVXOPT rather than quadprog.

    Returns
    -------
    x : numpy.array
        Solution that minimizes the quadratic problem.
    fval: double
        Value of the quadratic problem at the solution.
    exitflag: int
        Convergence flag of the problem. 1 for converged, 0 for failure.
    lambdav: dict
        Lagrange multipliers at the solution x. Contains the fields 'eq' and 'ineq' corresponding to the equality and
        inequality lagrange multipliers respectively.

    """
    # FIXME: (13/05/2019) CVXOPT behaving weirdly when using in caballero's algorithm, domain error. Changing to quadprog
    if solver == "cvxopt":  # CVXOPT
        H = matrix(H)
        f = matrix(f.reshape(-1).astype(np.double))

        if A.size == 0:
            A = None
            b = None
        else:
            A = matrix(A)
            b = matrix(b.reshape(-1).astype(np.double))

        if Aeq.size == 0:
            Aeq = None
            beq = None
        else:
            Aeq = matrix(Aeq)
            beq = matrix(beq.reshape(-1).astype(np.double))

        if x0.size == 0:
            x0 = None
        else:
            x0 = matrix(x0.reshape(-1).astype(np.double))

        if solver == "cvxopt" or solver is None:
            solver = None
        else:
            solver = "mosek"

        sol = cvxqp(H, f, A, b, Aeq, beq, solver=solver, initvals=None)

        x = np.array(sol['x']).flatten()
        fval = sol['primal objective']

        if sol['status'] == 'optimal':
            exitflag = 1
        else:
            exitflag = 0

        lambdav = {'eq': np.array(sol['y']).reshape(-1), 'ineq': np.array(sol['z']).reshape(-1)}

    elif solver == 'quadprog' or solver is None:
        meq = 0 if beq is None else beq.size

        if meq == 0:  # no equality constraints
            C_qp = -A.T
            b_qp = -b.flatten()
        else:
            C_qp = -np.vstack((Aeq, A)).T
            b_qp = -np.concatenate((beq.flatten(), b.flatten()))

        C_qp = None if C_qp.size == 0 else C_qp
        b_qp = None if b_qp.size == 0 else b_qp

        try:
            sol = solve_qp(H, -f.flatten(), C=C_qp, b=b_qp, meq=meq)
        except ValueError:  # QP failed
            x = x0 if x0 is not None else np.array([])
            fval = np.array([])
            exitflag = 0
            lambdav = {'eq': np.array([]), 'ineq': np.array([])}
        else:  # QP successful
            x, fval, _, _, lmbd, _ = sol
            exitflag = 1
            x = x.flatten()
            lmbd = lmbd.flatten()
            if meq != 0:
                active_set_idx = np.sort(sol[5] - 1)  # the -1 is to change from fortran index to c index
                lmbd_active_set = mldivide(C_qp[:, active_set_idx], H @ x + f.flatten())
            else:
                lmbd_active_set = np.array([])
            lambdav = {'eq': lmbd_active_set[:meq], 'ineq': lmbd[meq:]}

    return x, fval, exitflag, lambdav
