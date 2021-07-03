from abc import ABC, abstractmethod


class ProcedureOptions(ABC):
    """Base class for infill procedure options configuration."""

    _VALID_NLP_SOLVERS = ['ipopt']

    @property
    def max_fun_evals(self):
        """Maximum number of (rigorous) model function evaluations."""
        return self._max_fun_evals

    @max_fun_evals.setter
    def max_fun_evals(self, value):
        if isinstance(value, int):
            if value > 0:
                self._max_fun_evals = value
            else:
                raise ValueError("'max_fun_evals' has to be a positive "
                                 "integer.")

        else:
            raise ValueError("'max_fun_evals' has to be a integer.")

    @property
    def feasible_tol(self):
        """Feasibility tolerance for the model constraints g(x) <= tol."""
        return self._feasible_tol

    @feasible_tol.setter
    def feasible_tol(self, value):
        if isinstance(value, float):
            if value > 0.0:
                self._feasible_tol = value
            else:
                raise ValueError("'feasible_tol' has to be a positive small "
                                 "value.")

        else:
            raise TypeError("'feasible_tol' has to be a float.")

    # -------------------------------------------------------------------------

    def __init__(self, max_fun_evals: int = 500, feasible_tol: float = 1e-6):
        # initialize base class
        super().__init__()

        # class base parameters
        self.max_fun_evals = max_fun_evals
        self.feasible_tol = feasible_tol

    @abstractmethod
    def check_options_setup(self):
        """Abstract method to be overriden where the procedure options are
        checked before the optimization starts.
        """
        pass
