import numpy as np
from surropt.utils.matrixdivide import mrdivide

# TODO: (06/05/2019) test the SQP routine.
def sqp(objfun, x0, confun=None, lb=None, ub=None, options=None):
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
            - qpsolver: Which library to use the quadratic programming (QP) solver. Default is 'cvxopt'. Other options
                        are: 'quadprog' and a matlab engine object in order to use the quadprog routine from MATLAB.

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

    >>> x, fval, exitflag, nfunevals, lagrange_mult, iterations = sqp(objective, x0, constraints)

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

    # TODO: verify that the constraint function assertion if working
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

        p, qpfval, qpexflag, lambdadict = __qp_solve(B, c, -C, ci, -F, ce, x0=x, solver=qpsolver[0], engine=qpsolver[1])

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
    # end sqp


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

    d, ce = nonlcon(x)[:2]

    # call the merit function
    phi_x_mu, obj, objgrad, globalls = __merit_fun(obj, objgrad, objfun, nonlcon, x, mu, globalls)

    D_phi_x_mu = c.conj().T @ p  # directional derivative

    # only the elements of d corresponding to violated constraints should be included
    idx = d < 0
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
    ci, ce = nonlcon(x)[:2]

    idx = ci < 0  # indexes of negative (violated) constraints

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
        cif = np.vstack((cif_eval, x[lbidx].reshape(-1, 1) - lb[lbidx].reshape(-1, 1),
                         ub[ubidx].reshape(-1, 1) - x[ubidx].reshape(-1, 1)))
    else:
        cif = np.vstack((x[lbidx].reshape(-1, 1) - lb[lbidx].reshape(-1, 1),
                         ub[ubidx].reshape(-1, 1) - x[ubidx].reshape(-1, 1)))

        if cif.size == 0:  # boundless problem and without constraints
            cif = cif.reshape(-1, 1)  # to avoid concatenation problem

    if cigf_eval.size != 0:
        cigf = np.vstack((cigf_eval, bnds_grad))
    else:
        cigf = bnds_grad

    return cif, cef, cigf, cegf


# Check inputs auxiliaries
def __check_vector_input(x0, lb, ub):
    # if any of the bounds is set to None, make them a infinity vector
    if lb is None:
        lb = - np.ones((x0.size, 1)) * np.inf

    if ub is None:
        ub = np.ones((x0.size, 1)) * np.inf

    x0 = x0.reshape(-1, 1)  # force to be column
    lb = lb.reshape(-1, 1)
    ub = ub.reshape(-1, 1)

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
        tolstep = 1e-6
        tolopt = 1e-6
        tolcon = 1e-6
        qpsolver = ('cvxopt', None)

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
                qpsolver = (options['qpsolver'], None)
                if qpsolver[0] != 'cvxopt' or qpsolver[0] != 'quadprog':
                    ValueError("Invalid QP solver option. Valid values are 'cvxopt' and 'quadprog'.")
            else:  # not string, check for matlab engine
                if not isinstance(options['qpsolver'], matlab.engine.matlabengine.MatlabEngine):
                    ValueError('Invalid object for engine. You must specify a matlab engine object.')
                else:  # it is indeed a matlab engine, set default parameters
                    qpsolver = ('matlab', options['qpsolver'])

        else:
            qpsolver = ('cvxopt', None)

        return maxiter, maxfunevals, tolstep, tolopt, tolcon, qpsolver

    else:  # invalid argument
        ValueError("options argument has to be a dictionary.")


def __constraint_function_check(confun, x0):
    if callable(confun):  # if confun is a callable function
        cif_eval, cef_eval, cigf_eval, cegf_eval = confun(x0)

        # check if constraint functions are returning column vectors
        if not (cif_eval.ndim == 2 and cif_eval.shape[1] == 1 or cif_eval.shape[1] == 0) and cif_eval.size != 0:
            ValueError("Inequality constraint function must return a column array.")

        if not (cef_eval.ndim == 2 and cef_eval.shape[1] == 1 or cef_eval.shape[1] == 0) and cef_eval.size != 0:
            ValueError("Equality constraint function must return a column array.")

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
