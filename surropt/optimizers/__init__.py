import numpy as np
# import ipopt

from surropt.caballero.problem import CaballeroProblem
from surropt.optimizers.utils import __check_vector_input, __set_options_structure, __empty_nonlcon, __bnd2cf, \
    __qp_solve, __linesearch
from surropt.utils.matrixdivide import mrdivide


def optimize_nlp(obj_surr: dict, con_surr: list, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray, solver=None):
    """Optimization interface for several NLP solvers (e.g. IpOpt, SQP-active set, etc.).

    Parameters
    ----------
    obj_surr : dict
        Surrogate model structure of the objective function.
    con_surr : list
        List of surrogate models of the constraints functions.
    x0 : ndarray
        Initial estimate.
    lb : ndarray
        Lower bound of variables in the NLP problem.
    ub : ndarray
        Upper bound of variables in the NLP problem.
    solver : str
        Type of NLP solver to be used. Default is None, which corresponds to IpOpt solver.

    Returns
    -------
    out : tuple
        Tuple of 3 elements containg the optimal solution (first), objective function value at solution (second) and
        exit flag (third). The exit flag assumes two values: 0 for failure of convergence, 1 for success.
    """

    if solver is None or solver == 'ipopt':
        nlp = ipopt.problem(
            n=x0.size,
            m=len(con_surr),
            problem_obj=CaballeroProblem(obj_surr, con_surr),
            lb=lb,
            ub=ub,
            cl=-np.inf * np.ones(len(con_surr)),
            cu=np.zeros(len(con_surr))
        )

        # ipopt options
        nlp.addOption('tol', 1e-6)
        nlp.addOption('constr_viol_tol', 1e-6)
        nlp.addOption('hessian_approximation', 'limited-memory')
        nlp.addOption('print_level', 0)
        nlp.addOption('mu_strategy', 'adaptive')

        x, info = nlp.solve(x0)
        fval = info['obj_val']
        exitflag = info['status']

        if exitflag == 0 or exitflag == 1 or exitflag == 6:
            exitflag = 1  # ipopt succeded
        else:
            exitflag = 0  # ipopt failed. See IpReturnCodes_inc.h for complete list of flags

    else:
        NotImplementedError("NLP solver not implemented.")

    return x, fval, exitflag


# TODO: (06/05/2019) test the SQP routine.
# TODO: (07/05/2019) assert constraint to return arrays even when there is a single constraint
def sqp(objfun: callable, x0: np.ndarray, confun: callable=None, lb=None, ub=None, options: dict =None):
    """
    This SQP implementation is the one described by [1]_ and implemented in Octave optimization toolbox.

    Min         f(x)
    subject to:
                c_i(x) >= 0,  for i = 1, ..., n

    where f(x) and c_i(x) are the objective and constraint functions

    Parameters
    ----------
    objfun : callable
        Objective function. Its gradient must be evaluated inside this function and returned as 2-D column array.
        See notes.
    x0 : numpy.array
        Initial estimate. For now, this implementation of SQP doesn't handle initial infeasibility. Therefore, `x0` has
        to be feasible.
    confun : callable
        Constraints and their Jacobian evaluation, both inequality and equality. See notes.
    lb : numpy.array
        Lower bounds. Same number of elements as `x0`. If any of the bounds arguments are set as None (default), it
        means that all elements of that bound are set as positive(upper bound)/negative(lower bound) infinity 1-D array.
    ub : numpy.array
        Upper bounds. If any of the bounds arguments are set as None (default), it means that all elements of that
        bound are set as positive(upper bound)/negative(lower bound) infinity 1-D array.
    options : dict
        Dictionary containing fields with algorithm options. Valid options are:
            - maxiter : maximum number of iterations. Default is 400;
            - maxfunevals : maximum number of function evaluations. Default is 100 * `x0`.size;
            - tolopt : Optimality tolerance. Default is 1e-6;
            - tolcon : Constraint tolerance. Default is 1e-6;
            - tolstep : Minimum step size tolerance. Default is 1e-6;
            - qpsolver: Which library to use the quadratic programming (QP) solver. Default is 'cvxopt'.

    Returns
    -------
    x : numpy.array
        Local solution if `exitflag` is positive.
    fval : double
        Objective function value at the solution `x`.
    exitflag : int
        Convergence flag. Values are:
        - 1 : First-order optimality measure is less than `options`['tolopt'] and minimum constraint violation was less
              than `options`['tolcon'].
        - 2 : Step size is less than `options`['tolstep'] and minimum constraint violation was less than
              `options`['tolcon'].
        - 0 : Maximum number of iterations achieved.
        - -1 : BFGS update failed.
        - -2 : Step size is less than `options`['tolstep']. However there still is some significant constraint
               violation.
    nevals : int
        Number of objective function evaluation.
    lambda : dict
        Dictionary containing equality ('eq') and inequality ('ineq') Lagrange multipliers at the solution `x`.
    iterations : int
        Number of iterations performed by the algorithm.

    Raises
    ------
    ValueError
        If `x0`,`lb` or `ub` aren't numeric arrays with same number of elements.
        If `lb` is greater `ub` for any element.
        If an invalid option parameter is set, e.g. typo in name parameter or the parameter doesn't exist.
        If the `objfun` or `confun` was set improperly.

    Notes
    -----
        ** How to set the `objfun` object **

        The `objfun` object must return a tuple of two elements. The first being the objective function evaluation
        (float) and the second being a 2-D column array containing the gradient of objective function.

        ** How to set the `confun` object **
        The `confun` callable object must return a tuple of four elements. The first two elements are the evaluations
        of inequality and equality constraints, respectively, returned as 2-D column array. The third and fourth
        elements of this tuple are the Jacobian evaluation of equality and inequality constraints, respectively.
        If the problem has only equalities, you must set the first and third elements as 2-D empty array
        (numpy.array([[]]). If it only has inequalities, set the second and fourth elements as 2-D empty array.


    References
    ----------
    .. [1] Nocedal, J. and S. J. Wright. Numerical Optimization, Second Edition. Springer Series in Operations
           Research, Springer Verlag, 2006.

    Examples
    --------
    ** Equality constraint only and no bounds problem **

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

    >>> x, fval, exitflag, nfunevals, lagrange_mult, iterations = utils(objective, x0, constraints)

    The values are:

    - x = [-1.71714352  1.59570964  1.82724584 -0.76364308 -0.76364308]
    - fval = 0.05394985
    - exitflag = 2
    - nfunevals = 8
    - lagrange_mult['eq'] = [-0.04016275 0.03795778 -0.00522259]
    - iterations = 6


    """

    # check input vectors x0, lb, ub
    x0, lb, ub = __check_vector_input(x0, lb, ub)

    # check options dictionary
    maxiter, maxfunevals, tolstep, tolopt, tolcon, qpsolver = __set_options_structure(options, x0)

    # check constraints
    # check if any confun parameters is coming as None (withtout constraints) and make it a empty handle
    if confun is None:
        confun = __empty_nonlcon()

    # __constraint_function_check(confun, x0)

    lbgrad = np.eye(x0.size)
    ubgrad = - lbgrad

    lbidx = lb != - np.inf
    ubidx = ub != np.inf
    lbgrad = lbgrad[lbidx.flatten(), :]
    ubgrad = ubgrad[ubidx.flatten(), :]

    # Transform bounds into inequality constraints
    confun = lambda xv, fun=confun: __bnd2cf(xv, lbidx, ubidx, lb, ub, np.vstack((lbgrad, ubgrad)), fun)

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

    lambdav = 100 * np.ones((ce.size + ci.size, 1))

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
        t2 = np.all(ci >= 0)
        t3 = np.all(lambda_i >= 0)
        t4 = np.linalg.norm(lambdav * con)

        if t2 and t3 and np.max([t0, t1, t4]) < tolopt:
            # Problem has converged with all constraints being satisfied
            exitflag = 1
            break

        # Solve the QP subproblem to compute the search direction p
        lambda_old = lambdav.copy()  # Store old multipliers

        p, qpfval, qpexflag, lambdadict = __qp_solve(B, c, -C, ci, -F, ce, x0=x, solver=qpsolver)

        if qpexflag == 1:
            lambdav[:nr_f, 0] = lambdadict['eq']
            lambdav[nr_f:, 0] = lambdadict['ineq']
        else:
            lambdav = lambda_old.copy()

        # Perform linesearch
        x_new, alpha, obj_new, c_new, globalls = __linesearch(x, p, objfun, confun, lambdav, obj, c, globalls)

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
            if np.linalg.norm(np.vstack((ce_new, ci_new)), -np.inf) < tolcon:
                exitflag = 2  # Step size is too small and constraint violation
                # is less than minimum constraint violation
            else:
                exitflag = -2  # Step size is too small but there some significante
                # constraint violation

            break

        # Check for number of function evaluations
        if globalls['nevals'] > maxfunevals:
            exitflag = 0
            break

        delxt = delx.conj().T

        d1 = delxt @ B @ delx

        t1 = 0.2 * d1
        t2 = delxt @ y

        if t2 < t1:
            theta = np.asscalar(0.8 * d1 / (d1 - t2))
        else:
            theta = 1

        r = theta * y + (1 - theta) * B @ delx

        d2 = delxt @ r

        if d1 == 0 or d2 == 0:
            exitflag = -1  # BFGS Update failed
            break

        B = B - mrdivide((B @ delx @ delxt @ B), d1) + mrdivide((r @ r.conj().T), d2)

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

    x = x.reshape(1, -1)  # Force to be row vector
    return x, obj, exitflag, nevals, lambdav, iteration
    # end utils