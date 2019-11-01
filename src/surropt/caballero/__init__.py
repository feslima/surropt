import warnings
from copy import deepcopy

import numpy as np
from colorama import Fore, Style, deinit, init
from pydace import Dace
from scipy.linalg import norm

from ..core.nlp import optimize_nlp
from ..core.options.nlp import DockerNLPOptions, NLPOptions
from ..core.procedures import InfillProcedure
from ..core.utils import (get_samples_index, is_row_member,
                          point_domain_distance)
from .problem import CaballeroOptions, is_inside_hypercube


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
            # the np.argmin already handles multiple minima as first ocurrence
            return feas_idx[np.argmin(feas_obj)].item()

    def build_surrogate(self, optimize=False, x=None, f=None, g=None) -> None:
        """Builds the objective and constraints surrogates based on whether or
        not to optimize their hyperparameters.

        Parameters
        ----------
        optimize : bool, optional
            Whether or not optimize the hyperparameters, by default False.

        x : np.ndarray, optional
            The input data to be used. If None, use the initial values from
            construction.

        f : np.ndarray, optional
            The function objective data to be used. If None, use the initial
            values from construction.

        g : np.ndarray, optional
            The constraint data to be used. If None, use the initial values
            from construction.
        """
        x = self.x if x is None else x
        g = self.g if g is None else g
        f = self.f if f is None else f

        n = x.shape[1]
        q = g.shape[1]

        if optimize:
            # optimize hyperparameters

            # initial theta
            one = np.ones((n,))
            theta0 = one
            lob = 1e-12 * theta0
            upb = 1e2 * theta0

            surr_obj = Dace(regression=self.regression,
                            correlation='corrgauss')
            surr_obj.fit(S=x, Y=f, theta0=theta0, lob=lob, upb=upb)

            # constraints metamodel
            surr_con = []
            for i in range(q):
                obj_ph = Dace(regression=self.regression,
                              correlation='corrgauss')
                obj_ph.fit(S=x, Y=g[:, i], theta0=theta0, lob=lob, upb=upb)

                surr_con.append(obj_ph)

            # store models
            self.surr_obj = surr_obj
            self.surr_con = surr_con

        else:
            # do not optimize hyperparameters, use previous model data
            if not hasattr(self, 'surr_obj') or not hasattr(self, 'surr_con'):
                # no previous model found, use recursion to build it
                self.build_surrogate(optimize=True, x=x, g=g, f=f)

            else:
                # objective function
                self.surr_obj.fit(S=x, Y=f,
                                  theta0=self.surr_obj.theta.flatten())

                # constraints
                for idx, obj in enumerate(self.surr_con):
                    obj.fit(S=x, Y=g[:, idx], theta0=obj.theta.flatten())

    def optimize(self):
        super().optimize()

        # initialize counter and domains
        self.k, self.j = 1, 1
        lb, ub = self.lb, self.ub
        self.lbopt, self.hlb, self.dlb = deepcopy(lb), deepcopy(lb), \
            deepcopy(lb)
        self.ubopt, self.hub, self.dub = deepcopy(ub), deepcopy(ub), \
            deepcopy(ub)
        self.fun_evals, self.move_num, self.contract_num = 0, 0, 0

        # initial NLP solver estimate
        x0 = self.x[self.best_feas_idx, :].flatten()

        # create internal variables of caballero
        self._xlhs = deepcopy(self.x)
        self._gobs = deepcopy(self.g)
        self._fobs = deepcopy(self.f)

        # initialize xstar, gstar and fstar
        self._xstar = x0.reshape(1, -1)
        self._gstar = self._gobs[self.best_feas_idx, :].reshape(1, -1)
        self._fstar = self._fobs[self.best_feas_idx].flatten()

        # termination flag
        self.terminated = False
        # colored font print
        init()

        while not self.terminated:
            sol = optimize_nlp(procedure=self, x=self._xlhs, g=self._gobs,
                               f=self._fobs, nlp_options=self.nlp_options,
                               x0=x0.tolist(), lb=self.lbopt.tolist(),
                               ub=self.ubopt.tolist())
            xjk, fjk, exitflag = sol['x'], sol['fval'], sol['exitflag']

            if exitflag < 1:
                print(Fore.RED + np.array2string(xjk, precision=4,
                                                 separator='\t', sign=' '))
            else:
                print(Fore.RESET + np.array2string(xjk, precision=4,
                                                   separator='\t', sign=' '))

            if self.fun_evals >= self.options.max_fun_evals:
                warnings.warn("Maximum number of function evaluations "
                              "achieved!")

                # search feasible indexes
                feas_idx = self.search_best_feasible_index(self._gobs,
                                                           self._fobs)
                if feas_idx is not None:
                    report_str = self.get_results_report(index=feas_idx,
                                                         r=0.005, x=self._xlhs,
                                                         f=self._fobs,
                                                         fun_evals=self.fun_evals)
                    print(report_str)

                # break loop
                self.terminated = True

            if not is_row_member(xjk, self._xlhs):
                sampled_results = self.sample_model(xjk)

                # update sample data
                self._xlhs = np.vstack((self._xlhs, xjk))
                self._gobs = np.vstack((self._gobs, sampled_results['g']))
                self._fobs = np.append(self._fobs, sampled_results['f'])

                self.fun_evals += 1

            else:
                # couldn't improve from last iteration
                xjk = self.refine()
                x0 = deepcopy(xjk)

            # wheter or not to update kriging parameters after refinement phase
            optimize_hyp = True if self.j == 1 else False

            self.build_surrogate(x=self._xlhs, f=self._fobs, g=self._gobs,
                                 optimize=optimize_hyp)

            # Starting from xjk solve the NLP to get xj1k
            sol = optimize_nlp(procedure=self, x=self._xlhs, g=self._gobs,
                               f=self._fobs, nlp_options=self.nlp_options,
                               x0=xjk.tolist(), lb=self.lbopt.tolist(),
                               ub=self.ubopt.tolist())
            xj1k, fj1k, exitflag = sol['x'], sol['fval'], sol['exitflag']

            if point_domain_distance(xjk, xj1k, lb, ub) >= \
                    self.options.ref_tol:
                # there was improvement, keep going
                xjk = xj1k
                self.j += 1

            else:
                xjk = self.refine()
                x0 = deepcopy(xjk)

        deinit()

    def sample_model(self, x: np.ndarray):
        sampled_data = self.model_function(x)
        f = sampled_data['f']
        g = sampled_data['g']

        # TODO: implement other parameters capture in 'extras' key
        # TODO: output data sanitation (g, f, and extras)
        return {'g': g, 'f': f}

    def refine(self):
        # find best value inserted
        x_ins = self._xlhs[self.m:, :]
        g_ins = self._gobs[self.m:, :]
        f_ins = self._fobs[self.m:]

        best_idx = self.search_best_feasible_index(g_ins, f_ins)

        if best_idx is not None:
            self._xstar = np.vstack((self._xstar, x_ins[best_idx, :]))
            self._gstar = np.vstack((self._gstar, g_ins[best_idx, :]))
            self._fstar = np.append(self._fstar, f_ins[best_idx])

        else:
            # no best sampled value found in the inserted points, just insert
            # the best value in the initial sample (i.e. no improvement).
            # This block shouldn't be possible. The else is a "just in case".

            # TODO: perform a contraction on the best feasible point instead
            best_init = self.search_best_feasible_index(self.g, self.f)

            raise NotImplementedError("Perform a contraction move.")

        # select the best point to be centered
        best_star = self.search_best_feasible_index(self._gstar, self._fstar)

        xstark = self._xstar[best_star, :].flatten()

        if self.contract_num == 0:
            contract_factor = self.options.first_factor
        else:
            contract_factor = self.options.second_factor

        # refine the hypercube limits
        self.refine_hypercube(xstark, contract_factor)

        # search for best feasible point
        feas_idx = self.search_best_feasible_index(self._gobs,
                                                   self._fobs)

        # check for termination
        if point_domain_distance(self._xstar[-1, :],
                                 self._xstar[-2, :], self.lb, self.ub) <= \
                self.options.term_tol:

            xstark = self._xlhs[feas_idx, :].flatten()

            # check if at least a contraction was made
            if self.contract_num > 0:
                # if so, terminate the algorithm
                if feas_idx is not None:
                    report_str = self.get_results_report(index=feas_idx,
                                                         r=0.005,
                                                         x=self._xlhs,
                                                         f=self._fobs,
                                                         fun_evals=self.fun_evals)
                    print(report_str)

                # break loop
                self.terminated = True

            else:
                # perform a large contraction if no contraction done
                self.refine_hypercube(xstark, contract_factor=0.9999)

        # update move counter and reset iteration counter
        self.k += 1
        self.j = 1

        return xstark

    def refine_hypercube(self, xstark: np.ndarray, contract_factor: float):
        d_range = self.hub - self.hlb

        # check if xstark is at domain bound
        if is_inside_hypercube(xstark, self.dlb, self.dub):
            # inside original domain
            if is_inside_hypercube(xstark, self.hlb, self.hub) and \
                norm(self.hub - self.hlb) / norm(self.dub - self.dlb) >= \
                    self.options.contraction_tol:
                # its inside hypercube, center and contract
                self._perform_contraction(xstark, contract_factor, d_range)

                print(Fore.YELLOW + '------------- Contract! -------------\n')

            else:
                # its at hypercube limit, center and move
                self._perform_move(xstark, d_range)

                print(Fore.GREEN + '--------------- Move! ---------------\n')

        else:
            # at domain limit
            if is_inside_hypercube(xstark, self.hlb, self.hub) and \
                norm(self.hub - self.hlb) / norm(self.dub - self.dlb) >= \
                    self.options.contraction_tol:
                # its inside hypercube, center and contract
                self._perform_contraction(xstark, contract_factor, d_range)

                print(Fore.YELLOW + '------------- Contract! -------------\n')

            else:
                # its at hypercube limit, center and move
                self._perform_move(xstark, d_range)

                print(Fore.GREEN + '--------------- Move! ---------------\n')

        # update optimization bounds and adjust them if needed
        self.lbopt = deepcopy(self.hlb)
        self.ubopt = deepcopy(self.hub)

        floor_lb = np.less(self.lbopt, self.dlb)
        if np.any(floor_lb):
            self.lbopt[floor_lb] = self.dlb[floor_lb]

        ceil_ub = np.greater(self.ubopt, self.dub)
        if np.any(ceil_ub):
            self.ubopt[ceil_ub] = self.dub[ceil_ub]

    def _perform_contraction(self, xstark, contract_factor, d_range):
        red_factor = (1 - contract_factor) * d_range / 2
        self.hlb = xstark - red_factor
        self.hub = xstark + red_factor

        self.contract_num += 1

        # inserted points
        x_ins = self._xlhs[self.m:, :]
        g_ins = self._gobs[self.m:, :]
        f_ins = self._fobs[self.m:]

        # each contraction discards the points outside new hypercube
        idx = get_samples_index(x_ins, self.hlb, self.hub)

        self._xlhs = np.vstack((self.x, x_ins[idx, :]))
        self._gobs = np.vstack((self.g, g_ins[idx, :]))
        self._fobs = np.append(self.f, f_ins[idx])

    def _perform_move(self, xstark, d_range):
        red_factor = d_range / 2
        self.hlb = xstark - red_factor
        self.hub = xstark + red_factor

        self.move_num += 1
