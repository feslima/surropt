from surropt.utils import distribute_data, build_surrogate
from tests_ import RESOURCES_PATH
import numpy as np


data = np.genfromtxt(RESOURCES_PATH / 'evap5k.csv', delimiter=',', skip_header=1)
data = data[:50, :]
options = {'input_index': [0, 1, 2, 3],
           'input_lb': [8.5, 0, 102, 0],
           'input_ub': [20., 100, 400, 400],
           'obj_index': 16,
           'con_index': [7, 10],
           'con_lb': [35.5, 40.],
           'con_ub': [np.inf, 80.],
           'con_tol': 1e-6,
           'reg_model': 'poly1',
           'cor_model': 'corrgauss'
           }

a, b, c = distribute_data(data, options)


d, e = build_surrogate(data[:, options['input_index']], np.hstack((a, b, c)), options)
