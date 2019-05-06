import numpy as np
import ipopt

from surropt.caballero.problem import CaballeroProblem


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
