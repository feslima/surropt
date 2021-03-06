import numpy as np
import pytest
from scipy.io import loadmat

from surropt.utils.models import evaporator
from surropt.caballero import Caballero
from tests_ import RESOURCES_PATH


# fixture
@pytest.fixture
def load_mat_file():
    mat_contents = loadmat(RESOURCES_PATH / "evap53pts.mat")
    return mat_contents['doeBuild']


def test_caballero(load_mat_file):
    # load input design
    x = load_mat_file[:, 3:7]

    # constraints
    x2_raw = load_mat_file[:, 10]
    p2_raw = load_mat_file[:, 13]

    g1 = 35.5 - x2_raw
    g2 = p2_raw - 80.0
    g3 = 40.0 - p2_raw

    g = np.c_[g1, g2, g3]

    # objective function
    f = load_mat_file[:, 19]

    # sampling function
    model_function = evaporator

    # bounds
    lb = [8.5, 0, 102, 0]
    ub = [20., 100., 400., 400.]

    caballero_obj = Caballero(x=x, g=g, f=f, model_function=model_function,
                              lb=lb, ub=ub, regression='poly1')

    caballero_obj.optimize()
    pass
