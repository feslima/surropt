import logging

import numpy as np

from surropt.optimizers.utils import (__bnd2cf, __check_vector_input,
                                      __constraint_function_check,
                                      __empty_nonlcon, __linesearch,
                                      __objective_function_check, __qp_solve,
                                      __set_options_structure)
from surropt.utils.matrixdivide import mrdivide
from tests_ import OPTIMIZERS_PATH


# module variables
LOG_OFF = False  # flag to turn on loggin of iterations

# logger configuration
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(levelname)s - %(message)s"
logging.basicConfig(filename=OPTIMIZERS_PATH / "sqp_log.log",
                    level=LOG_LEVEL,
                    format=LOG_FORMAT,
                    filemode='w')
logger = logging.getLogger()


def sqp(objfun: callable, x0: np.ndarray, confun: callable = None, lb=None,
        ub=None, options: dict = None):
    """
    This SQP implementation is the one described by [1]_ and implemented in
    Octave optimization toolbox.

    Min         f(x)
    subject to:
                c_i(x) <= 0,  for i = 1, ..., n

    where f(x) and c_i(x) are the objective and constraint functions

    Parameters
    ----------
    objfun : callable
        Objective function. Its gradient must be evaluated inside this function
        and returned as 2-D column array. See notes.

    x0 : numpy.array
        Initial estimate. For now, this implementation of SQP doesn't handle
        initial infeasibility. Therefore, `x0` hasto be feasible.

    confun : callable
        Constraints and their Jacobian evaluation, both inequality and
        equality. See notes.

    lb : {None, numpy.array}
        Lower bounds. Same number of elements as `x0`. If any of the bounds
        arguments are set as None (default), it means that all elements of that
        bound are set as positive(upper bound)/negative(lower bound) infinity
        1-D array.

    ub : {None, numpy.array}
        Upper bounds. If any of the bounds arguments are set as None (default),
        it means that all elements of that bound are set as positive(upper
        bound)/negative(lower bound) infinity 1-D array.

    options : dict
        Dictionary containing fields with algorithm options. Valid options are:

            - maxiter : maximum number of iterations. Default is 400;
            - maxfunevals : maximum function evaluations. Default is
                            100 * `x0`.size;
            - tolopt : Optimality tolerance. Default is 1e-6;
            - tolcon : Constraint tolerance. Default is 1e-6;
            - tolstep : Minimum step size tolerance. Default is 1e-6;
            - qpsolver: Which library to use the quadratic programming (QP)
                        solver. Default is 'quadprog'.

    Returns
    -------
    sol : dict
        Solution dictionary containing the fields:
        'x' : numpy.array
            Local solution if `exitflag` is positive.
        'fval' : double
            Objective function value at the solution `x`.
        'exitflag' : int
            Convergence flag. Values are:
            - 1 : First-order optimality measure is less than
                 `options`['tolopt'] and minimum constraint violation was less
                  than `options`['tolcon'].
            - 2 : Step size is less than `options`['tolstep'] and minimum
                  constraint violation was less than `options`['tolcon'].
            - 0 : Maximum number of iterations achieved.
            - -1 : BFGS update failed.
            - -2 : Step size is less than `options`['tolstep']. However there
                   still is some significant constraint violation.
        'nfevals' : int
            Number of objective function evaluation.
        'lambda' : dict
            Dictionary containing equality ('eq'), inequality ('ineq') and
            bounds ('lb', 'ub') Lagrange multipliers at the solution `x`.
        'iterations' : int
            Number of iterations performed by the algorithm.

    Raises
    ------
    ValueError
        If `x0`,`lb` or `ub` aren't numeric arrays with same number of elements
        If `lb` is greater `ub` for any element.
        If an invalid option parameter is set, e.g. typo in name parameter or
        the parameter doesn't exist.
        If the `objfun` or `confun` was set improperly.

    Notes
    -----
        ** How to set the `objfun` function **

        The `objfun` object must return a tuple of two elements. The first
        being the objective function evaluation (float) and the second being a
        2-D column array containing the gradient of objective function.

        ** How to set the `confun` function **
        The `confun` callable object must return a tuple of four elements. The
        first two elements are the evaluations of inequality and equality
        constraints, respectively, returned as 2-D column array. The third and
        fourth elements of this tuple are the Jacobian evaluation of equality
        and inequality constraints, respectively. If the problem has only
        equalities, you must set the first and third elements as 2-D empty
        array (numpy.array([[]]). If it only has inequalities, set the second
        and fourth elements as 2-D empty array.


    References
    ----------
    .. [1] Nocedal, J. and S. J. Wright. Numerical Optimization, Second
        Edition. Springer Series in Operations Research, Springer Verlag, 2006.

    Examples
    --------
    ** Single equality constraint, single inequality constraint and 4 bounds **

    Define the objective function as:

    >>> import numpy as np
    >>> def objective(x):
    ... # objective function - HS071 test problem
    ... obj = x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
    ...
    ... # gradient
    ... obj_grad = np.zeros((4,))
    ... obj_grad[0] = x[0] * x[3] + x[3]*(x[0]+x[1]+x[2])
    ... obj_grad[1] = x[0] * x[3]
    ... obj_grad[2] = x[0] * x[3] + 1
    ... obj_grad[3] = x[0]*(x[0]+x[1]+x[2])
    ...
    ... return obj, obj_grad

    Constraints:

    >>> def constraints(x):
    ...
    ... # define the inequality constraints
    ... ineq = np.array([[25 - x[0]*x[1]*x[2]*x[3]]])
    ... # define the inequality constraints Jacobian
    ... ineq_grad = np.zeros((1, 4))
    ... ineq_grad[0, 0] = -x[1] * x[2] * x[3]
    ... ineq_grad[0, 1] = -x[0] * x[2] * x[3]
    ... ineq_grad[0, 2] = -x[0] * x[1] * x[3]
    ... ineq_grad[0, 3] = -x[0] * x[1] * x[2]
    ...
    ... # define the equality constraints
    ... eq = np.array([[np.sum(np.multiply(x, x)) - 40]])
    ... # define the equality constraints Jacobian
    ... eq_grad = np.zeros((1, 4))
    ... eq_grad[0, 0] = 2 * x[0]
    ... eq_grad[0, 1] = 2 * x[1]
    ... eq_grad[0, 2] = 2 * x[2]
    ... eq_grad[0, 3] = 2 * x[3]
    ...
    ... return ineq, eq, ineq_grad, eq_grad

    Initial estimate

    >>> x0 = np.array([1., 5., 5., 1.])

    Call the solver

    >>> sol = sqp(objective, x0, constraints)
    >>> x = sol['x']
    >>> fval = sol['fval']
    >>> exitflag = sol['exitflag']
    >>> nfunevals = sol['nfevals']
    >>> lagrange_mult = sol['lambda']
    >>> iterations = sol['iterations']

    The values are:

    - x = [1. 4.74299964 3.82114998 1.37940829]
    - fval = 17.014017289178593
    - exitflag = 2
    - nfunevals = 7
    - lagrange_mult['eq'] = [-0.16146857]
    - lagrange_mult['ineq'] = [-0.55229366]
    - lagrange_mult['lb'] = [-1.08787123e+00 -3.40273704e-12 -8.67542702e-12 -5.52639122e-12]
    - lagrange_mult['ub'] = [-3.46930334e-12 -6.12161682e-10 -8.34873225e-12 -3.39974223e-12]
    - iterations = 5


    ** Three equality constraints, no inequalities and no bounds problem **

    Define the objective function as:

    >>> import numpy as np
    >>> def objective(x):
    ... # objective function
    ... obj = np.exp(np.prod(x)) - 0.5*(x[0]**3 + x[1]**3 + 1)**2
    ...
    ... # gradient
    ... gradobj = np.zeros((5, 1))
    ... gradobj[0] = (np.prod(x[1:]) * np.exp(np.prod(x)) - (3*x[0]**2)*(x[0]**3+x[1]**3+1))
    ... gradobj[1] = np.prod(x[np.arange(len(x)) != 1]) * np.exp(np.prod(x)) - (3*x[1]**2)*(x[0]**3+x[1]**3+1)
    ... gradobj[2] = np.prod(x[np.arange(len(x)) != 2]) * np.exp(np.prod(x))
    ... gradobj[3] = np.prod(x[np.arange(len(x)) != 3]) * np.exp(np.prod(x))
    ... gradobj[4] = np.prod(x[np.arange(len(x)) != 4]) * np.exp(np.prod(x))
    ...
    ... return obj, gradobj

    Constraints:

    >>> def constraints(x):
    ...
    ... # define the equality constraints
    ... r = np.array([
    ...     [np.sum(x**2) - 10],
    ...     [x[1]*x[2] - 5*x[3]*x[4]],
    ...     [x[0]**3 + x[1]**3 + 1]
    ... ])
    ... # define the equality constraints Jacobian
    ... rgrad = np.zeros((3, 5))
    ... rgrad[0, 0] = 2 * x[0]
    ... rgrad[0, 1] = 2 * x[1]
    ... rgrad[0, 2] = 2 * x[2]
    ... rgrad[0, 3] = 2 * x[3]
    ... rgrad[0, 4] = 2 * x[4]
    ...
    ... rgrad[1, 0] = 0
    ... rgrad[1, 1] = x[2]
    ... rgrad[1, 2] = x[1]
    ... rgrad[1, 3] = -5 * x[4]
    ... rgrad[1, 4] = -5 * x[3]
    ...
    ... rgrad[2, 0] = 3 * x[0] ** 2
    ... rgrad[2, 1] = 3 * x[1] ** 2
    ... rgrad[2, 2] = 0
    ... rgrad[2, 3] = 0
    ... rgrad[2, 4] = 0
    ...
    ... return np.array([[]]), r, np.array([[]]), rgrad

    Initial estimate

    >>> x0 = np.array([-1.8, 1.7, 1.9, -0.8, -0.8])

    Call the solver

    >>> sol = sqp(objective, x0, constraints)
    >>> x = sol['x']
    >>> fval = sol['fval']
    >>> exitflag = sol['exitflag']
    >>> nfunevals = sol['nfevals']
    >>> lagrange_mult = sol['lambda']
    >>> iterations = sol['iterations']

    The values are:

    - x = [-1.71714355  1.59570967  1.82724579 -0.76364308 -0.76364308]
    - fval = 0.05394984777028391
    - exitflag = 2
    - nfunevals = 9
    - lagrange_mult['eq'] = [-0.04016274  0.03795777 -0.00522264]
    - lagrange_mult['ineq'] = []
    - lagrange_mult['lb'] = []
    - lagrange_mult['ub'] = []
    - iterations = 7


    """

    if LOG_OFF:
        logger.disabled = True

    # check input vectors x0, lb, ub
    x0, lb, ub = __check_vector_input(x0, lb, ub)

    logger.debug("Initial estimate (x0) - " +
                 np.array2string(x0, precision=4, separator=',',
                                 suppress_small=True))
    logger.debug("Iter #\t|\tx_i\t|\tp_i\t|\tfval")

    # check the objective function
    __objective_function_check(objfun, x0)

    # check options dictionary
    maxiter, maxfunevals, tolstep, tolopt, tolcon, qpsolver = \
        __set_options_structure(options, x0)

    # check if any confun parameters is coming as None (withtout constraints)
    # and make it a empty handle
    if confun is None:
        confun = __empty_nonlcon()

    # check constraints
    __constraint_function_check(confun, x0)

    lbgrad = - np.eye(x0.size)
    ubgrad = - lbgrad

    lbidx = lb != - np.inf
    ubidx = ub != np.inf
    lbgrad = lbgrad[lbidx, :]
    ubgrad = ubgrad[ubidx, :]

    # Transform bounds into inequality constraints
    def confun(xv, fun=confun): return __bnd2cf(xv, lbidx, ubidx, lb, ub,
                                                np.vstack((lbgrad, ubgrad)),
                                                fun)

    # global structure for parametrization
    globalls = {'nevals': 0}

    # Initialize variables: objective and constraints
    x = x0

    obj, c = objfun(x0)
    globalls['nevals'] = 1

    # Initialize the positive definite Hessian Approximation
    n = x0.size
    B = np.eye(n)

    # Evaluate the constraint info
    ci, ce, C, F = confun(x0)

    # Jacobian Matrix of bothe equalites and inequalites
    A = np.vstack((F, C))

    # Choose an initial lambda

    lambdav = 100 * np.ones((ce.size + ci.size,))

    iteration = 0

    while iteration < maxiter:
        # Convergence check
        nr_f = ce.size  # Number of equality constraints

        # Split lagrange multipliers
        lambda_i = lambdav[nr_f:]

        con = np.vstack((ce, ci))

        # Karush-Kuhn-Tucker conditions

        t0 = np.linalg.norm(c - A.conj().T @ lambdav)
        t1 = np.linalg.norm(ce)
        t2 = np.all(ci <= 0)
        t3 = np.all(lambda_i >= 0)
        t4 = np.linalg.norm(lambdav * con)

        if t2 and t3 and np.max([t0, t1, t4]) < tolopt:
            # Problem has converged with all constraints being satisfied
            exitflag = 1
            break

        # Solve the QP subproblem to compute the search direction p
        lambda_old = lambdav.copy()  # Store old multipliers

        p, qpfval, qpexflag, lambdadict = __qp_solve(
            B, c, C, -ci, F, -ce, x0=np.array([]), solver=qpsolver)

        if qpexflag == 1:
            lambdav[:nr_f] = -lambdadict['eq']
            lambdav[nr_f:] = -lambdadict['ineq']
        else:
            lambdav = lambda_old.copy()
            p = x

        # Perform linesearch
        x_new, alpha, obj_new, c_new, globalls = __linesearch(
            x, p, objfun, confun, lambdav, obj, c, globalls)

        # Re-evaluate objective, constraints and gradients at new x value
        ci_new, ce_new, C_new, F_new = confun(x_new)

        A_new = np.vstack((F_new, C_new))

        y = c_new - c

        if A.size != 0:
            t = (A_new - A).T @ lambdav
            y -= t

        delx = x_new - x  # Step size

        # Check if step size is too small
        if np.linalg.norm(delx) < tolstep * np.linalg.norm(x):
            # Check for minimum constraint violation
            if np.vstack((ce_new, ci_new)).size == 0 or \
                    np.max(np.vstack((ce_new, ci_new))) < tolcon:
                # the first verification is for cases when there are no
                # constraints, so there is no exception when using the max

                # Step size is too small and constraint violation is less than
                # minimum constraint violation
                exitflag = 2
            else:
                # Step size is too small but there some significant constraint
                # violation
                exitflag = -2

            break

        # Check for number of function evaluations
        if globalls['nevals'] > maxfunevals:
            exitflag = 0
            break

        delxt = delx[:, np.newaxis].conj().T

        # the item is to guarantee that the result is a scalar
        d1 = (delxt @ B @ delx).item()

        t1 = 0.2 * d1
        t2 = delxt @ y

        if t2 < t1:
            theta = np.asscalar(0.8 * d1 / (d1 - t2))
        else:
            theta = 1

        r = theta * y + (1 - theta) * B @ delx

        d2 = (delxt @ r).item()

        if d1 == 0 or d2 == 0:
            exitflag = -1  # BFGS Update failed
            break

        # the new axis is to regularize the algebraic vector operations
        # (r and delx are column vectors)
        B = B - mrdivide(B @ delx[:, np.newaxis] @ delxt @ B, d1) + \
            mrdivide(r[:, np.newaxis] @ r[:, np.newaxis].conj().T, d2)

        str_fmt = "{0} |\t{1} |\t{2} |\t{3}"
        x_str = np.array2string(x.flatten(), precision=4,
                                separator=',', suppress_small=True)
        p_str = np.array2string(p.flatten(), precision=4,
                                separator=',', suppress_small=True)
        logger.debug(str_fmt.format(iteration, x_str, p_str, obj))

        # Update new values
        x = x_new
        obj = obj_new
        c = c_new
        ce = ce_new
        F = F_new
        ci = ci_new
        C = C_new
        A = A_new

        iteration += 1

    if iteration >= maxiter:
        exitflag = 0  # Too many iterations

    nevals = globalls['nevals']

    n_c_eq = ce.size
    n_bnds = 0 if np.all(lb == -np.inf) and np.all(ub == np.inf) else lb.size
    n_c_iq = ci.size - 2 * n_bnds
    lmbda_dict = {'eq': lambdav[:n_c_eq],
                  'ineq': lambdav[n_c_eq:(n_c_eq + n_c_iq)],
                  'lb': lambdav[(n_c_eq + n_c_iq):(n_c_eq + n_c_iq + n_bnds)],
                  'ub': lambdav[(n_c_eq + n_c_iq + n_bnds): (n_c_eq + n_c_iq + 2 * n_bnds)]}

    return {'x': x, 'fval': obj, 'exitflag': exitflag, 'lambda': lmbda_dict,
            'iterations': iteration, 'nfevals': nevals}
    # end sqp
