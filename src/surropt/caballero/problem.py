import os
from string import Template

import numpy as np
from colorama import Fore, Style, deinit, init

from ..core.options import ProcedureOptions
from ..core.procedures.output import Report

__all__ = ["CaballeroOptions", "is_inside_hypercube", "CaballeroReport"]


class CaballeroReport(Report):
    """Report class specifically for caballero procedure only.
    """

    def __init__(self, terminal=False, plot=False):
        super().__init__(terminal=terminal, plot=plot)

    def build_iter_report(self, movement: str, iter_count: int, x: list,
                          f_pred: float, f_actual: float, g_actual: float,
                          header=False, field_size: int = 12):
        arr_str = super().build_iter_report(iter_count=iter_count, x=x,
                                            f_pred=f_pred, f_actual=f_actual,
                                            g_actual=g_actual, header=header,
                                            field_size=field_size)
        mv_temp = Template("{:$f_size}\t").substitute(f_size=field_size)

        # concatentate movement and iteration string report
        if header:
            mv_str = 'Last move'
        else:
            mv_str = movement

        return mv_temp.format(mv_str) + arr_str

    def print_iteration(self, movement: str, iter_count: int, x: list,
                        f_pred: float, f_actual: float, g_actual: float,
                        header=False, field_size: int = 12, color_font=None):

        arr_str = self.build_iter_report(movement=movement,
                                         iter_count=iter_count, x=x,
                                         f_pred=f_pred, f_actual=f_actual,
                                         g_actual=g_actual, header=header,
                                         field_size=field_size)

        if self.terminal:
            # terminal asked, check font color
            if color_font == 'red':
                print(Fore.RED + arr_str)
            else:
                print(Fore.RESET + arr_str)


class CaballeroOptions(ProcedureOptions):
    """Options structure for the Caballero class algorithm.

    Parameters
    ----------
    max_fun_evals : int, optional
        Maximum number of black box function evaluations, by default 500.

    feasible_tol : float, optional
        Feasibility tolerance for the model constraints g(x) <= tol, by default
        1e-06.

    penalty_factor : float, optional
        Value to penalize the objective function when its value returned by the
        black box model indicates that is a infeasible result (note that by
        infeasible it is referring to whether or not the sampling converged for
        that case, not optimization feasibility), by default None.

        See this property notes for more info.

    ref_tol : float, optional
        Refinement tolerance specification (tol1), by default 1e-4.

    term_tol : float, optional
        Termination tolerance specification (tol2), by default 1e-5. Has to be
        lesser than `ref_tol`.

    first_factor : float, optional
        First contraction factor specification, by default 0.6. Has to be
        between 0 and 1.

    second_factor : float, optional
        Subsequent contraction factor specification, by default 0.4. Has to be
        between 0 and 1. Has to be lesser than `first_factor`.

    contraction_tol : float, optional
        Maximum contraction size that the refinement hypercube can achieve
        when compared to the original domain, by default 1e-4.

        See this property notes for more information.
    """
    @property
    def penalty_factor(self):
        """Value to penalize the objective function when its value returned by
        the black box model indicates that is a infeasible result (note that
        "infeasible result" it is referring to whether or not the sampling
        converged for that case, not constraint feasibility), by default None.

        The value set for the penalty factor is simply summed to the objective
        function value in order to make the optimization procedure avoid
        sampling in the known infeasible regionself.

        If this value is set to None, the procedure will automatically chose
        the highest value of the objective function in the initial sampling as
        the penalty factor."""
        return self._penalty_factor

    @penalty_factor.setter
    def penalty_factor(self, value):
        self._penalty_factor = value

    @property
    def first_factor(self):
        """First contraction factor specification."""
        return self._first_factor

    @first_factor.setter
    def first_factor(self, value):
        if isinstance(value, float):
            if 0.0 < value < 1.0:
                self._first_factor = value
            else:
                raise ValueError("'first_factor' value has to be in the "
                                 "range (0, 1).")

        else:
            raise ValueError("'first_factor' has to be a float.")

    @property
    def second_factor(self):
        """Second contraction factor specification."""
        return self._second_factor

    @second_factor.setter
    def second_factor(self, value):
        if isinstance(value, float):
            if 0.0 < value < 1.0:
                self._second_factor = value
            else:
                raise ValueError("'second_factor' value has to be in the "
                                 "range (0, 1).")

        else:
            raise ValueError("'second_factor' has to be a float.")

    @property
    def ref_tol(self):
        """Refinement tolerance specification (tol1)."""
        return self._ref_tol

    @ref_tol.setter
    def ref_tol(self, value):
        if isinstance(value, float):
            if value > 0.0:
                self._ref_tol = value
            else:
                raise ValueError("'ref_tol' value has to be a positive float.")

        else:
            raise ValueError("'ref_tol' has to be a float.")

    @property
    def term_tol(self):
        """Termination tolerance specification (tol2)."""
        return self._term_tol

    @term_tol.setter
    def term_tol(self, value):
        if isinstance(value, float):
            if value > 0.0:
                self._term_tol = value
            else:
                raise ValueError("'term_tol' value has to be a positive "
                                 "float.")

        else:
            raise ValueError("'term_tol' has to be a float.")

    @property
    def contraction_tol(self):
        """Maximum contraction size that the refinement hypercube can achieve
        when compared to the original domain.

        The ratio between the current refined hypercube range and the original
        domain range can't be lesser than `contraction_tol`. After the ratio
        reaches this value, no more contractions will be perfomed."""
        return self._contraction_tol

    @contraction_tol.setter
    def contraction_tol(self, value):
        if isinstance(value, float):
            if value > 0.0:
                self._contraction_tol = value
            else:
                raise ValueError("'contraction_tol' value has to be a positive"
                                 " float.")

        else:
            raise ValueError("'contraction_tol' has to be a float.")

    # -------------------------------------------------------------------------
    def __init__(self, max_fun_evals: int = 500, feasible_tol: float = 1e-06,
                 penalty_factor: float = None,
                 ref_tol: float = 1e-4, term_tol: float = 1e-5,
                 first_factor: float = 0.6, second_factor: float = 0.4,
                 contraction_tol: float = 1e-4):

        super().__init__(max_fun_evals=max_fun_evals,
                         feasible_tol=feasible_tol)

        self.penalty_factor = penalty_factor
        self.first_factor = first_factor
        self.second_factor = second_factor
        self.ref_tol = ref_tol
        self.term_tol = term_tol
        self.contraction_tol = contraction_tol

        # perform instance checkup
        self.check_options_setup()

    def check_options_setup(self):
        if self.first_factor <= self.second_factor:
            raise ValueError("'first_factor' has to be greater than "
                             "'second_factor'.")

        if self.ref_tol <= self.term_tol:
            raise ValueError("'ref_tol' has to be greater than "
                             "'term_tol'.")


def is_inside_hypercube(point: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                        tol: float = 1e-8):
    """Determines if a `point` is inside a hypercube following a specified
    tolerance.

    Parameters
    ----------
    point : np.ndarray
        Point to be checked (1D array).

    lb : np.ndarray
        Hypercube lower bound (1D array).

    ub : np.ndarray
        Hypercube upper bound(1D array).

    tol : float, optional
        Tolerance value to accept whether or not `point` is at the limit.
        Default is 1e-8. (This value is a percentage of domain range.)
    """

    # Offset the hypercube to the origin
    lbf = np.zeros(lb.shape)
    ubf = ub - lb

    # offset the point in relation to the origin as well
    pointf = point - lb

    # check if the point is outside the hypercube. If so, raise exception
    if np.any(np.greater(pointf, ubf)) or np.any(np.less(pointf, lbf)):

        def arr2str_fcn(x): return np.array2string(x, precision=4,
                                                   separator='\t', sign=' ')

        err_msg = ("Point: {0}\nLB: {1}\nUB: {2}".format(arr2str_fcn(point),
                                                         arr2str_fcn(lb),
                                                         arr2str_fcn(ub)))
        raise ValueError("The point is outside the hypercube.\n" + err_msg)
    else:
        # check if the point is inside or at limit
        if np.any(np.abs((ubf - pointf) / ubf) <= tol) \
                or np.any(np.abs((pointf - lbf) / ubf <= tol)):
            # limit
            return False
        else:
            # inside
            return True
