import numpy as np
import scipy.io as sio
from surropt.caballero import caballero
from tests_ import RESOURCES_PATH
from tests_.models.evaporator_model import evaporator_doe

lb = [8.5, 0., 102., 0.]
ub = [20., 100., 400., 400.]
mv_index = list(range(3, 7))
fobj_index = 19
const_index = [10, 13]
const_tol = 1e-6
const_lb = [35.5, 40.]
const_ub = [np.inf, 80.]
tol1 = 1e-5
tol2 = 1e-6
first_contract = 0.7
second_contract = 0.2
max_fun_evals = 200

coptions = {'input_lb': lb, 'input_ub': ub,
            'input_index': mv_index,
            'obj_index': fobj_index,
            'con_index': const_index,
            'con_tol': const_tol,
            'con_lb': const_lb, 'con_ub': const_ub,
            'reg_model': 'poly1', 'cor_model': 'corrgauss',
            'tol1': tol1, 'tol2': tol2, 'max_fun_evals': 200,
            'first_contract': first_contract, 'second_contract': second_contract,
            'nlp_solver': 'sqp', 'penalty_factor': 1e3}

# load the initial sampling
mat_contents = sio.loadmat(RESOURCES_PATH / 'evap53pts.mat')

doe_build = mat_contents["doeBuild"]
# mat_contents = sio.loadmat("coptions_test.mat")
# doe_build = mat_contents["doeBuild"]

caballero(doe_build, evaporator_doe, coptions)
