from copy import deepcopy
from colorama import Fore, Style, init, deinit

import numpy as np
from pydace import Dace

from ..core.procedures import InfillProcedure
from .problem import CaballeroOptions
from ..core.options.nlp import NLPOptions, DockerNLPOptions
from ..core.nlp import optimize_nlp


class Caballero(InfillProcedure):
    def __init__(self, x: np.ndarray, g: np.ndarray, f: np.ndarray,
                 model_function, lb: np.ndarray, ub: np.ndarray,
                 regression: str, options: CaballeroOptions = None,
                 nlp_options: NLPOptions = None):
        # kriging regression model
        self.regression = regression

        # proceed with default options for caballero procedure if none defined
        options = CaballeroOptions() if options is None else options

        # proceed with default options for NLP solver as docker server
        nlp_options = DockerNLPOptions(name='docker-server',
                                       server_url='http://192.168.99.100:5000') \
            if nlp_options is None else nlp_options

        # initialize mother class
        super().__init__(x=x, g=g, f=f, model_function=model_function, lb=lb,
                         ub=ub, options=options, nlp_options=nlp_options)

        # perform a setup check
        self.check_setup()

        # initial surrogate construction
        self.build_surrogate(optimize=True)

    def check_setup(self):
        # perform basic checkup
        super().check_setup()

        # search for feasible points in the initial sampling
        best_feas_idx = self.search_best_feasible_index(g=self.g, f=self.f)

        if best_feas_idx is None:
            raise IndexError("No initial feasible point found. You need at "
                             "least one feasible case in the sample.")
        else:
            self.best_feas_idx = best_feas_idx

    def search_best_feasible_index(self, g: np.ndarray, f: np.ndarray) -> int:
        # get feasible points indexes for a given set of samples
        feas_idx = np.nonzero(np.all(g <= self.options.feasible_tol,
                                     axis=1))[0]

        if feas_idx.size == 0:
            # no feasible point found, return None
            return None
        else:
            # get feasible objective function values
            feas_obj = f[feas_idx]

            # get best feasible index
            return feas_idx[np.argmin(feas_obj)].item()

    def build_surrogate(self, optimize=False) -> None:
        """Builds the objective and constraints surrogates based on whether or
        not to optimize their hyperparameters.

        Parameters
        ----------
        optimize : bool, optional
            Whether or not optimize the hyperparameters, by default False.

        """
        n = self.x.shape[1]
        q = self.g.shape[1]

        if optimize:
            # optimize hyperparameters

            # initial theta
            one = np.ones((n,))
            theta0 = one
            lob = 1e-12 * theta0
            upb = 1e2 * theta0

            surr_obj = Dace(regression=self.regression,
                            correlation='corrgauss')
            surr_obj.fit(S=self.x, Y=self.f, theta0=theta0, lob=lob, upb=upb)

            # constraints metamodel
            surr_con = []
            for i in range(q):
                obj_ph = Dace(regression=self.regression,
                              correlation='corrgauss')
                obj_ph.fit(S=self.x, Y=self.g[:, i],
                           theta0=theta0, lob=lob, upb=upb)

                surr_con.append(obj_ph)

            # store models
            self.surr_obj = surr_obj
            self.surr_con = surr_con

        else:
            # do not optimize hyperparameters, use previous model data
            if not hasattr(self, 'surr_obj') or not hasattr(self, 'surr_con'):
                # no previous model found, use recursion to build it
                self.build_surrogate(optimize=True)

            else:
                # objective function
                self.surr_obj.fit(S=self.x, Y=self.f,
                                  theta0=self.surr_obj.theta.flatten())

                # constraints
                for obj in self.surr_con:
                    obj.fit(S=self.x, Y=self.f, theta0=obj.theta.flatten())

    def optimize(self):
        super().optimize()

        # initialize counter and domains
        k, j = 1, 1
        lb, ub = self.lb, self.ub
        lbopt, hlb, dlb = deepcopy(lb), deepcopy(lb), deepcopy(lb)
        ubopt, hub, dub = deepcopy(ub), deepcopy(ub), deepcopy(ub)
        fun_evals, move_num, contract_num = 0, 0, 0

        # initial NLP solver estimate
        x0 = self.x[self.best_feas_idx, :].flatten()

        # colored font print
        init()

        while True:
            sol = optimize_nlp(procedure=self, nlp_options=self.nlp_options,
                               x0=x0.tolist(), lb=lbopt.tolist(),
                               ub=ubopt.tolist())
            xjk, fjk, exitflag = sol['x'], sol['fval'], sol['exitflag']

            if exitflag <= 0:
                print(Fore.RED + np.array2string(xjk, precision=4,
                                                 separator='\t', sign=' '))
            else:
                print(Fore.RESET + np.array2string(xjk, precision=4,
                                                   separator='\t', sign=' '))

            fun_evals += 1

            if fun_evals >= self.options.max_fun_evals:
                break

        deinit()
