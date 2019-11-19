import json
from abc import ABC, abstractmethod

import numpy as np
import requests

from ..options.nlp import DockerNLPOptions, NLPOptions
from ..procedures import InfillProcedure


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

        dic = {'x0': x0,
               'lb': lb,
               'ub': ub,
               'surr_data': surr_data}

        # send data to server and return the nlp solution
        response = requests.post(url=nlp_options.server_url + '/opt', json=dic)
        sol = response.json()
        sol['x'] = np.array(sol['x'])  # converting from list to np.array

    else:
        raise NotImplementedError("NLP solver not implemented.")

    return sol
