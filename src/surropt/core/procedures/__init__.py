from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.linalg import norm
from scipy.spatial import cKDTree

from ..options import ProcedureOptions
from ..options.nlp import NLPOptions
from ..utils import _is_numeric_array_like
from .output import Report


class InfillProcedure(ABC):
    """Base class where the optimization through infill criteria is done.
    """

    @property
    def x(self) -> np.ndarray:
        """Design input samples."""
        return self._x

    @x.setter
    def x(self, value):
        if _is_numeric_array_like(value):
            self._x = np.asarray(value, dtype=float)
        else:
            raise ValueError("'x' has to be a numeric array.")

    @property
    def m(self) -> int:
        """Number of initial samples. (READONLY)"""
        return self._m

    @property
    def g(self) -> np.ndarray:
        """Observed constraint data."""
        return self._g

    @g.setter
    def g(self, value):
        if _is_numeric_array_like(value):
            # TODO: ensure it is not empty
            self._g = np.asarray(value, dtype=float)
        else:
            raise ValueError("'g' has to be a numeric array.")

    @property
    def f(self) -> np.ndarray:
        """Observed objective function data."""
        return self._f

    @f.setter
    def f(self, value):
        if _is_numeric_array_like(value):
            self._f = np.asarray(value, dtype=float).flatten()

        else:
            raise ValueError("'f' has to be a numeric array.")

    @property
    def model_function(self):
        """Rigorous model where the data will be sampled."""
        return self._model_function

    @model_function.setter
    def model_function(self, value):
        if callable(value) or value is None:
            self._model_function = value
        else:
            raise ValueError("'model_function' has to be a callable object.")

    @property
    def options(self):
        """Procedure options setup"""
        return self._options

    @options.setter
    def options(self, value: ProcedureOptions):
        if isinstance(value, ProcedureOptions):
            self._options = value
        else:
            raise ValueError("'options' has to be a valid options structure.")

    @property
    def nlp_options(self):
        """Non-Linear Programming solver options."""
        return self._nlp_options

    @nlp_options.setter
    def nlp_options(self, value: NLPOptions):
        if isinstance(value, NLPOptions):
            self._nlp_options = value
        else:
            raise ValueError("'nlp_options' has to be a valid NLP option "
                             "instance.")

    @property
    def lb(self):
        """Input design lower bound."""
        return self._lb

    @lb.setter
    def lb(self, value):
        if _is_numeric_array_like(value):
            self._lb = np.asarray(value, dtype=float).flatten()

        else:
            raise ValueError("'lb' has to be a numeric array.")

    @property
    def ub(self):
        """Input design upper bound."""
        return self._ub

    @ub.setter
    def ub(self, value):
        if _is_numeric_array_like(value):
            self._ub = np.asarray(value, dtype=float).flatten()

        else:
            raise ValueError("'ub' has to be a numeric array.")

    @property
    def report_options(self):
        """Optimization procedure options report (i.e. plot or return strings
        containing info about the procedure iterations)"""
        return self._report_options

    @report_options.setter
    def report_options(self, value):
        if isinstance(value, Report):
            self._report_options = value
        else:
            raise ValueError("'report_options' has to be a 'Report' object.")

    # -------------------------------------------------------------------------
    def __init__(self, x: np.ndarray, g: np.ndarray, f: np.ndarray,
                 model_function, lb: np.ndarray, ub: np.ndarray,
                 options: ProcedureOptions, nlp_options: NLPOptions,
                 report_options: Report):
        super().__init__()

        self.x = x
        self.g = g
        self.f = f
        self.model_function = model_function
        self.lb = lb
        self.ub = ub
        self.options = options
        self.nlp_options = nlp_options
        self.report_options = report_options

    @abstractmethod
    def check_setup(self):
        """ Checks if the problem data is ready to be optimized (i.e. check
        input, constraints and objective array dimensions, etc.)
        """
        # check input shapes
        n_x, d_x = self.x.shape
        n_g, d_g = self.g.shape
        n_f = self.f.size

        if self.f.ndim != 1:
            raise ValueError("The objective function sampled data has be a 1D "
                             "array.")

        if n_x != n_g or n_x != n_f:
            raise ValueError("'x', 'g' and 'f' must have the same number of "
                             "rows.")

        # if input dimensions are ok, store the number of initial points
        self._m = n_x

        d_lb = self.lb.size
        d_ub = self.ub.size

        if d_lb != d_ub:
            raise ValueError("'lb' and 'ub' must have the same size.")

        if d_x != d_lb or d_x != d_ub:
            raise ValueError("The number of input dimensions ('x') has to be "
                             "the same as 'lb' and 'ub'.")

        if np.any(self.lb >= self.ub):
            raise ValueError("'lb' elements must be greater than 'ub'.")

        # options checkup
        self.options.check_options_setup()

    @abstractmethod
    def optimize(self):
        """Performs the infill criteria optimization.
        This is an abstract method. Must be overriden.
        """

        self.check_setup()
