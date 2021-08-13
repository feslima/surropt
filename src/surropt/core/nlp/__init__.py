import json
from abc import ABC, abstractmethod

import cyipopt
import numpy as np
import requests
from pydace import Dace

from ..options.nlp import DockerNLPOptions, IpOptOptions, NLPOptions
from ..procedures import InfillProcedure
from ..utils import CaballeroProblem


def optimize_nlp(procedure: InfillProcedure, x: np.ndarray, g: np.ndarray,
                 f: np.ndarray, nlp_options: NLPOptions, x0: list, lb: list,
                 ub: list):
    if not isinstance(procedure, InfillProcedure):
        raise ValueError("'procedure' has to be a valid Infill procedure "
                         "object.")

    if not isinstance(nlp_options, NLPOptions):
        raise ValueError("'nlp_options' has to be a valid Non-Linear options "
                         "object.")

    if isinstance(nlp_options, DockerNLPOptions):
        # use docker server

        # constraint hyperparameters
        con_theta = [model.theta.flatten().tolist()
                     for model in procedure.surr_con]

        # process data to be sent
        surr_data = {'input_design': x.tolist(),
                     'fobj_data': {
            'fobj_obs': f.tolist(),
            'fobj_theta': procedure.surr_obj.theta.flatten().tolist()},
            'const_data': {
            'const_obs': g.tolist(),
            'const_theta': con_theta},
            'regmodel': procedure.regression,
            'corrmodel': 'corrgauss'
        }

        # ipopt options
        nlp_opts = {
            'tol': nlp_options.tol,
            'max_iter': nlp_options.max_iter,
            'con_tol': nlp_options.con_tol
        }

        dic = {'x0': x0,
               'lb': lb,
               'ub': ub,
               'surr_data': surr_data,
               'nlp_opts': nlp_opts}

        # send data to server and return the nlp solution
        response = requests.post(url=nlp_options.server_url + '/opt', json=dic)
        sol = response.json()
        sol['x'] = np.array(sol['x'])  # converting from list to np.array

    elif isinstance(nlp_options, IpOptOptions):
        # local installation of IpOpt solver
        x0 = np.array(x0).flatten()
        lb = np.array(lb).flatten()
        ub = np.array(ub).flatten()

        m, n = x.shape

        # hyperparameters
        f_theta = procedure.surr_obj.theta.flatten()
        con_theta = [model.theta.flatten() for model in procedure.surr_con]

        # build the surrogates
        obj_surr = Dace(regression=procedure.regression,
                        correlation='corrgauss')
        obj_surr.fit(S=x, Y=f, theta0=f_theta)

        con_surr = []
        for j in range(g.shape[1]):
            con_surr_ph = Dace(regression=procedure.regression,
                               correlation='corrgauss')
            con_surr_ph.fit(S=x, Y=g[:, j],
                            theta0=con_theta[j])
            con_surr.append(con_surr_ph)

        # ------------------------------ Solver call ------------------------------
        nlp = cyipopt.Problem(
            n=x0.size,
            m=len(con_surr),
            problem_obj=CaballeroProblem(obj_surr, con_surr),
            lb=lb,
            ub=ub,
            cl=-np.inf * np.ones(len(con_surr)),
            cu=np.zeros(len(con_surr))
        )

        # ipopt options
        nlp.add_option('tol', nlp_options.tol)
        nlp.add_option('constr_viol_tol', nlp_options.con_tol)
        nlp.add_option('max_iter', nlp_options.max_iter)
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('print_level', 0)
        nlp.add_option('mu_strategy', 'adaptive')

        x, info = nlp.solve(x0)
        fval = info['obj_val']
        exitflag = info['status']

        if exitflag == 0 or exitflag == 1 or exitflag == 6:
            # ipopt succeded
            exitflag = 1
        else:
            # ipopt failed. See IpReturnCodes_inc.h for complete list of flags
            exitflag = 0

        return {'x': x, 'fval': fval, 'exitflag': exitflag}

    else:
        raise NotImplementedError("NLP solver not implemented.")

    return sol
