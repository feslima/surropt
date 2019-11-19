import numpy as np
import pytest
from scipy.io import loadmat

from src.surropt.utils.models import evaporator
from tests_ import RESOURCES_PATH


# fixture
@pytest.fixture
def load_mat_file():
    mat_contents = loadmat(RESOURCES_PATH / "evap53pts.mat")
    return mat_contents['doeBuild']


def test_evaporator_model(load_mat_file):
    check = []

    n_samples = load_mat_file.shape[0]

    for i in range(n_samples):
        # inputs
        f1_lhs, f3_lhs, p100_lhs, f200_lhs = load_mat_file[i][3:7].tolist()

        # results from matlab
        res = dict(zip(["f2", "f4", "f5", "x2", "t2", "t3", "p2", "f100",
                        "t100", "q100", "t201", "q200"],
                       load_mat_file[i][7:19].tolist()))

        # python model call
        sol = evaporator(f1=f1_lhs, f3=f3_lhs, p100=p100_lhs, f200=f200_lhs)

        check.append(all(np.allclose(res[k], sol['extras'][k])
                         for k in res.keys()))

    assert all(check), "Evaporator model failed in some cases."
