import os
from abc import ABC, abstractmethod

import numpy as np
from colorama import Fore, Style, deinit, init
from scipy.linalg import norm
from scipy.spatial import cKDTree


class Report(ABC):
    # TODO: write class documentation
    def __init__(self, terminal=False, plot=False):
        super().__init__()

        self.terminal = terminal
        self.plot = plot

    @property
    def terminal(self):
        """Whether or not to print each iteration info as string."""
        return self._terminal

    @terminal.setter
    def terminal(self, value):
        if isinstance(value, bool):
            self._terminal = value
        else:
            raise ValueError("'terminal' property only accepts True or False.")

    @property
    def plot(self):
        """Whether or not to plot the iteration process."""
        return self._plot

    @plot.setter
    def plot(self, value):
        if isinstance(value, bool):
            self._plot = value
        else:
            raise ValueError("'plot' property only accepts True or False.")

    def print_iteration(self, iter_count: int, x: list, f_pred: float,
                        f_actual: float, g_actual: float, header=False,
                        color_font=None):
        n_x = len(x)

        if header:
            mv_header = [" x" + str(i + 1) for i in range(n_x)]
            str_arr = ['Movement', 'Iter'] + mv_header + \
                ['f_pred', 'f_actual', 'feasibility']
            arr_str = ("{:10}\t"*len(str_arr)).format(*str_arr)
            # arr_str = "{0:^10s}".format(''.join(map(str, str_arr)))

        else:
            i = str(iter_count)
            mv_arr = np.array(x)
            num_arr = np.append(x, np.array([f_pred, f_actual, g_actual]))
            formatter = {'float_kind': lambda x: '{0: 10.4e}'.format(x)}
            str_arr = np.array2string(num_arr, separator='\t',
                                      max_line_width=os.get_terminal_size()[0],
                                      formatter=formatter)[1:-1]
            mov = 'test'  # placeholder variable for movement type
            arr_str = "{0:10}\t{1:10}\t{2}".format(mov, i, str_arr)

        if self.terminal:
            # terminal print asked, check for font color
            if color_font == 'red':
                print(Fore.RED + arr_str)
            else:
                print(Fore.RESET + arr_str)

        return arr_str

    def plot_iteration(self):
        if self.plot:
            raise NotImplementedError("plot iteration not implemented!")

    def get_results_report(self, index: int, r: float, x: np.ndarray,
                           f: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                           fun_evals: int) -> str:
        """Returns the results message report that contains info about the
        neighbourhood of `x` inside a specified domain (`lb` and `ub`).

        Parameters
        ----------
        index : int
            Row index of `x` that corresponds to the optimal point.
        r : float
            Radius percentage of the domain euclidian range (`lb` and `ub`) to
            search points near `x`.
        x : np.ndarray
            Sample input variables to be analysed (2D array).
        f : np.ndarray
            Sample observed objective function values (1D array, number of
            elements has to be the same as the number of rows in `x`).
        lb : np.ndarray
            Domain lower bound of the sample `x`. It is assumed that ALL the
            row of `x` are inside this domain.
        ub : np.ndarray
            Domain upper bound of the sample `x`. It is assumed that ALL the
            row of `x` are inside this domain.
        fun_evals : int
            Number of function evaluations needed to obtain the sample `x`

        Returns
        -------
        str
            Results report message to be printed in the terminal/cmd.
        """
        # search nearest points within r euclidian distance
        kdtree = cKDTree(data=x)
        euc_dom_rng = norm(ub - lb, ord=2)
        neigh_idx = kdtree.query_ball_point(x=x[index, :], r=r*euc_dom_rng)
        results_msg = ("\nBest feasible value found: {0:8.4f} at point\n"
                       "x = {1}\n"
                       "{2} points are within {3:.3%} euclidian range of this "
                       "point based on original domain.\n"
                       "Number of function evaluations: {4}")
        num_arr = np.array2string(x[index, :], precision=4, separator='\t',
                                  sign=' ')[1:-1]
        results_msg = results_msg.format(f[index],
                                         num_arr,
                                         len(neigh_idx),
                                         r, fun_evals)

        if self.terminal:
            # terminal asked, print directly
            print(Fore.RESET + results_msg)

        return results_msg
