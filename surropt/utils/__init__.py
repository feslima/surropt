import numpy as np
from pydace import dacefit


def distribute_data(doe_data: np.ndarray, options: dict):
    """Sets up the data into proper variables for surrogate construction. The constrains are rearranged to c_i(x) < 0

    Parameters
    ----------
    doe_data : ndarray
        Design of experiments data array containing input and output data from model. The rows are the cases and columns
        are the variables.
    options : dict
        Dictionary containing info about the `doe_data` array (i.e. input, objective function and constraints indexes).

    Returns
    -------
    out : tuple
        Tuple of three elements. The first one is the input data of the experiment. The second is the objective
        function. The third is the reformulated constraint data (c(x) < 0)
    """

    # perform a key check in options dictionary
    if not all(v in options for v in ['input_index', 'obj_index', 'con_index']):
        raise KeyError("Index data (either input, objective or constraint) not present as key in option dictionary.")
    else:
        inp_idx = np.asarray(options['input_index']).flatten()
        obj_idx = np.asarray(options['obj_index']).flatten()
        con_idx = np.asarray(options['con_index']).flatten()

    if con_idx.size == 0:  # no constraints specified
        con_idx = []  # to prevent slicing exceptions
        con_data = doe_data[:, con_idx]

    else:
        # sanity check of constraint limit data
        if not all(v in options for v in ['con_lb', 'con_ub']):
            raise KeyError("Constraint limit data missing (either lower or upper bounds)")
        else:  # constraint bounds
            con_lb = np.asarray(options['con_lb']).flatten()
            con_ub = np.asarray(options['con_ub']).flatten()

            if con_lb.size != con_ub.size:
                raise ValueError("Constraint bound data (lower and upper) must have the same size.")

            if con_idx.size != con_lb.size:
                raise ValueError("Number of constraint indexes must be the same as limits specified")

            if (con_lb >= con_ub).any():  # check if constraint bounds are properly set
                raise ValueError("All constraints lower bounds must be lower than their upper bounds.")

            # preallocate
            con_data = doe_data[:, con_idx]

            # classify the indexes on types 0, 1 or 2
            # 0 for       -Inf <= c_i(x) <= some value
            # 1 for some value <= c_i(x) <= Inf
            # 2 for some value <= c_i(x) <= some value
            type_idx = np.full(con_idx.shape, np.nan)
            has_type_0, has_type_1, has_type_2 = False, False, False
            for i in range(con_idx.size):
                if con_lb[i] == -np.inf and con_ub[i] != np.inf:  # type 0
                    type_idx[i] = 0
                    has_type_0 = True
                elif con_lb[i] != -np.inf and con_ub[i] == np.inf:  # type 1
                    type_idx[i] = 1
                    has_type_1 = True
                else:
                    type_idx[i] = 2
                    has_type_2 = True

            if has_type_0:
                type0_con_index = np.nonzero(type_idx == 0)[0]
                con_data[:, type0_con_index] = con_data[:, type0_con_index] - con_ub[type0_con_index]

            if has_type_1:
                type1_con_index = np.nonzero(type_idx == 1)[0]
                con_data[:, type1_con_index] = con_lb[type1_con_index] - con_data[:, type1_con_index]

            if has_type_2:
                # duplicate the indexes positions
                type2_con_index = np.nonzero(type_idx == 2)[0]
                lb_data = con_lb[type2_con_index] - con_data[:, type2_con_index]
                ub_data = con_data[:, type2_con_index] - con_ub[type2_con_index]

                con_data[:, type2_con_index] = lb_data
                con_data = np.insert(con_data, type2_con_index + 1, ub_data, axis=1)

    inp_data = doe_data[:, inp_idx]
    obj_data = doe_data[:, obj_idx]

    return inp_data, obj_data, con_data


def build_surrogate(input_data: np.ndarray, obs_data: np.ndarray, options: dict, opt_hyper=[]):
    """Builds a surrogate for each response present in `obs_data`.

    Parameters
    ----------
    input_data: ndarray
        Input data from design of experiments.
    obs_data: ndarray
        Response (output) data from design of experiments.
    options: dict
        Dictionary containing the regression and correlation model specifications.
    opt_hyper: list
        Whether or not to optimize the hyperparameters of the surrogate model. Default is empty (perform optimization).
        If the hyperparameters are specified (non-empty list) the surrogates are built without performing optimization.
        The specification format should be a m-by-n list where m is the number of input dimensions and n is the number
        of output dimensions.

    Returns
    -------
    out: tuple
        Tuple of two elements. The first is the list of surrogate models, the second is the performance list of the
        models.
    """

    # sanity check
    if 'reg_model' not in options:
        raise KeyError("Regression model not specified in options dictionary.")
    else:
        reg_model = options['reg_model']

    if 'cor_model' not in options:
        raise KeyError("Correlation model not specified in options dictionary.")
    else:
        cor_model = options['cor_model']

    if not isinstance(opt_hyper, list):
        raise TypeError("opt_hyper has to be a list object.")
    else:
        opt_hyper = np.asarray(opt_hyper)

        if opt_hyper.ndim == 2:
            m, n = opt_hyper.shape

            if m != input_data.shape[1]:
                raise ValueError("The number of dimensions of the hyperparameters  has to be the same as the input "
                                 "array.")

            if n != obs_data.shape[1]:
                raise ValueError("The number of hyperparameters  has to be the same as the output array.")

    _, ndim_des = input_data.shape
    theta0 = np.ones((ndim_des,))
    lb_theta = 1e-12 * theta0
    ub_theta = 1e2 * theta0

    _, ndim_sur = obs_data.shape

    krmodel = []
    perf = []
    if opt_hyper.size == 0:  # empty list, perform optimization
        for i in range(ndim_sur):
            krPH, perfPH = dacefit(input_data, obs_data[:, i], reg_model, cor_model, theta0, lob=lb_theta, upb=ub_theta)
            krmodel.append(krPH)
            perf.append(perfPH)

    else:  # hyperparameters specified, do not perform optimization
        for i in range(ndim_sur):
            krPH, perfPH = dacefit(input_data, obs_data[:, i], reg_model, cor_model, opt_hyper[:, i])
            krmodel.append(krPH)
            perf.append(perfPH)

    return krmodel, perf


def sample_model(x, fun_handle: callable, options: dict, args=()):
    # sample the model
    sample = fun_handle(x, *args)

    if not all(k in sample for k in ['solution', 'status']):
        raise KeyError("Solution dictionary returned from the sample function must have the keys 'solution' and "
                       "'status'")

    # distribute data for surrogate
    x_samp, fobs_sampled, gobs_sampled = distribute_data(sample['solution'], options)
    x_samp = x_samp.flatten()
    fobs_sampled = fobs_sampled.item()
    gobs_sampled = gobs_sampled.flatten()

    # penalize the objective function if the sample has not converged
    penalty_factor = options['penalty_factor']
    fobs_sampled = fobs_sampled if sample['status'] else fobs_sampled + penalty_factor

    return x_samp, fobs_sampled, gobs_sampled
