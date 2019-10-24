import numpy as np
from scipy.io import loadmat

from surropt.utils.models import evaporator
from surropt.caballero import Caballero
from tests_ import RESOURCES_PATH
from surropt.core.options.nlp import DockerNLPOptions


def main():
    mat_contents = loadmat(RESOURCES_PATH / "evap53pts.mat")
    mat_contents = mat_contents['doeBuild']

    # load input design
    x = mat_contents[:, 3:7]

    # constraints
    x2_raw = mat_contents[:, 10]
    p2_raw = mat_contents[:, 13]

    g1 = 35.5 - x2_raw
    g2 = p2_raw - 80.0
    g3 = 40.0 - p2_raw

    g = np.c_[g1, g2, g3]

    # objective function
    f = mat_contents[:, 19]

    # sampling function
    def model_function(x):
        return evaporator(f1=x[0], f3=x[1], p100=x[2], f200=x[3])

    # bounds
    lb = [8.5, 0, 102, 0]
    ub = [20., 100., 400., 400.]

    # nlp server options
    nlp_opts = DockerNLPOptions(name='wsl-server',
                                server_url='http://localhost:5000')

    caballero_obj = Caballero(x=x, g=g, f=f, model_function=model_function,
                              lb=lb, ub=ub, regression='poly1',
                              nlp_options=nlp_opts)

    caballero_obj.optimize()


if __name__ == "__main__":
    main()
