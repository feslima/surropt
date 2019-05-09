import numpy as np
import scipy.io as sio
from surropt.caballero import caballero
from tests_ import RESOURCES_PATH


lb = [8.5, 0., 102., 0.]
ub = [20., 100., 400., 400.]
mv_index = list(range(3, 7))
fobj_index = 19
const_index = [10, 13]
const_tol = 1e-6
const_lb = [35.5, 40.]
const_ub = [100., 80.]
tol1 = 1e-4
tol2 = 1e-5
max_fun_evals = 200

coptions = {'input_lb': lb, 'input_ub': ub,
            'input_index': mv_index,
            'obj_index': fobj_index,
            'con_index': const_index,
            'con_tol': const_tol,
            'con_lb': const_lb, 'con_ub': const_ub,
            'reg_model': 'poly1', 'cor_model': 'corrgauss',
            'tol1': tol1, 'tol2': tol2, 'max_fun_evals': 200,
            'nlp_solver': 'sqp'}


# load the initial sampling
mat_contents = sio.loadmat(RESOURCES_PATH / 'evap53pts.mat')

doe_build = mat_contents["doeBuild"]
# mat_contents = sio.loadmat("coptions_test.mat")
# doe_build = mat_contents["doeBuild"]

caballero(doe_build, coptions)