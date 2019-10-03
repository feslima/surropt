import numpy as np
from ..core.procedures import InfillProcedure
from .problem import CaballeroOptions


class Caballero(InfillProcedure):
    def __init__(self, x: np.ndarray, g: np.ndarray, f: np.ndarray,
                 model_function, lb: np.ndarray, ub: np.ndarray,
                 options: CaballeroOptions = None):

        # proceed with default options for caballero procedure
        options = CaballeroOptions() if options is None else options

        # initialize mother class
        super().__init__(x, g, f, model_function, lb, ub, options)

    def check_setup(self):
        # perform basic checkup
        super().check_setup()

        # search for feasible points in the initial sampling
        raise NotImplementedError("Best feasible search not implemented.")

    def search_best_feasible_index(self):
        pass

    def build_surrogate(self):
        pass

    def optimize(self):
        super().optimize()
