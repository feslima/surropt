from surropt.utils import distribute_data, build_surrogate
import numpy as np
import pathlib


parent_path = pathlib.Path(__file__).parent.parent

data = np.genfromtxt(parent_path / 'resources\\evap5k.csv', delimiter=',', skip_header=1)
data = data[:50, :]
options = {'input_index': [0, 1, 2, 3],
           'obj_index': 16,
           'con_index': [7, 10],
           'con_lb': [35.5, 40.],
           'con_ub': [np.inf, 80.],
           'reg_model': 'poly1',
           'cor_model': 'corrgauss'
           }

a, b, c = distribute_data(data, options)


d, e = build_surrogate(data[:, options['input_index']], np.hstack((a, b, c)), options)
