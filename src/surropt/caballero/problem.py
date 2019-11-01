import numpy as np

from ..core.options import ProcedureOptions

__all__ = ["CaballeroOptions", "is_inside_hypercube"]


class CaballeroOptions(ProcedureOptions):
    # TODO: Document caballero procedure options __init__

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
        when compared to the original domain."""
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
                 nlp_solver: str = 'ipopt', penalty_factor: float = None,
                 ref_tol: float = 1e-4, term_tol: float = 1e-5,
                 first_factor: float = 0.6, second_factor: float = 0.4,
                 contraction_tol: float = 1e-4):

        super().__init__(max_fun_evals=max_fun_evals,
                         feasible_tol=feasible_tol,
                         nlp_solver=nlp_solver)

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
        [description]
    lb : np.ndarray
        [description]
    ub : np.ndarray
        [description]
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
