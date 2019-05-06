import numpy as np
import scipy.spatial as spatial

from surropt.utils import distribute_data, build_surrogate
from surropt.optimizers import optimize_nlp


def caballero(doe_initial: np.ndarray, options: dict):
    # status table:
    # 0 - converged successfully
    # 1 - maximum number of function evaluations achieved and a feasible point was found

    if 'tol1' not in options.keys():
        raise KeyError("First contraction tolerance specification not found in options dictionary.")
    else:
        tol1 = options['tol1']

    if 'tol2' not in options.keys():
        raise KeyError("Second contraction tolerance specification not found in options dictionary.")
    else:
        tol2 = options['tol2']

    if 'con_tol' not in options.keys():
        raise KeyError("Constraint tolerance specification not found in options dictionary.")
    else:
        con_tol = options['con_tol']

    if 'max_fun_evals' not in options.keys():
        raise KeyError("Maximum function evaluations specification not found in options dictionary.")
    else:
        max_fun_evals = options['max_fun_evals']

    if not all([v in options.keys() for v in ['input_lb', 'input_ub']]):
        raise KeyError("Either lower or upper bounds for input variables is not specified.")
    else:
        lb = np.asarray(options['input_lb']).flatten()
        ub = np.asarray(options['input_ub']).flatten()

        if lb.size != ub.size:
            raise ValueError("Lower and upper bound arrays must have the same number of elements.")

        if (lb >= ub).any():
            raise ValueError("All values for lower bounds must be lower than their upper bounds.")

    n_initial_samples, _ = doe_initial.shape

    # shape the data
    x_samp, fobs, gobs = distribute_data(doe_initial, options)

    # set the iterations counter
    k, j = 0, 0

    # build the surrogate
    surr_models, _ = build_surrogate(x_samp, np.hstack((fobs, gobs)), options)
    obj_model = surr_models[0]
    con_model = surr_models[1:]

    # search for a feasible point in the initial sampling
    best_sampled_idx = np.nonzero(fobs == np.min(fobs[np.all(gobs <= options['con_tol'], axis=1), 0]))[0]
    if best_sampled_idx.size == 0:
        raise ValueError("No feasible point found in the initial sampling. It is advised to implement a search criteria"
                         " before proceeding with optimization procedure.")

    x0 = x_samp[best_sampled_idx, :].flatten()
    xjk = x0.copy()

    # Initialization of the refinement procedure
    lbopt, hlb, dlb = lb, lb, lb
    ubopt, hub, dub = ub, ub, ub
    move_num = 0
    contract_num = 0
    fun_evals = 0

    while True:
        xjk, fjk, exitflag = optimize_nlp(obj_model, con_model, xjk, lbopt, ubopt, solver=None)

        # check for maximum number of function evaluations
        if fun_evals >= max_fun_evals:
            if fobs[np.all(gobs <= con_tol, axis=1), 0].size != 0:
                feasible_index = np.nonzero(fobs == np.min(fobs[np.all(gobs <= con_tol, axis=1), 0]))[0]
                point_tree = spatial.cKDTree(x_samp)
                idx = np.asarray((point_tree.query_ball_point(x_samp[feasible_index, :],
                                                              0.005 * np.linalg.norm(dlb - dub)))[0])
                status = 1

            break

        if not xjk.reshape(-1).tolist() in x_samp.tolist():  # if the last solution found is not repeated
            # sample the point
            raise NotImplementedError("Sampling not implemented")

        fun_evals += 1

        if k != 0 and j == 0:
            surr_models, _ = build_surrogate(x_samp, np.hstack((fobs, gobs)), options, opt_hyper=[])
            obj_model = surr_models[0]
            con_model = surr_models[1:]
        else:  # do not update kriging parameters, just insert points
            opt_hyper = np.zeros((xjk.size, 1 + len(con_model)))
            opt_hyper[:, 0] = obj_model['theta'].flatten()
            for i in range(1, len(con_model)):
                opt_hyper[:, i] = con_model[i]['theta'].flatten()

            # FIXME: (05/05/2019) not debugged. Check opt_hyper and models returned from build_surrogate
            surr_models, _ = build_surrogate(x_samp, np.hstack((fobs, gobs)), options, opt_hyper=opt_hyper)
            obj_model = surr_models[0]
            con_model = surr_models[1:]

        # Starting from xjk solve the NLP to get xj1k
        xj1k, fj1k, exitflag = optimize_nlp(obj_model, con_model, xjk, lbopt, ubopt, solver=None)

        if np.linalg.norm(xjk - xj1k) / np.linalg.norm(dlb - dub) >= tol1:
            xjk = xj1k
            j += 1

        else:
            raise NotImplementedError("Refinement procedure not implemented.")