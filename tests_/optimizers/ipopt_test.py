import numpy as np
import scipy.io as sio
import json
import ipopt

from pydace import dacefit
from tests_ import RESOURCES_PATH
from surropt.caballero.problem import CaballeroProblem


# -------------------------- Store as JSON a numpy.ndarray or any nested-list composition --------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


# -------------------------- load doe data to construct dacefit models and dump them as json --------------------------
dump_json = False
mat_contents = sio.loadmat(RESOURCES_PATH / "column_doe.mat")

MV = mat_contents['MV']
CV = mat_contents['CV']

con = np.hstack((-CV[:, [4]], CV[:, [2, 5]]))

ones = np.ones((MV.shape[1],))
theta0 = ones
lb = 1e-3 * ones
ub = 1e3 * ones

krmodel = []
perf = []
for i in np.arange(con.shape[1]):
    krmodelPH, perfPH = dacefit(MV, con[:, i], 'poly1', 'corrgauss', theta0, lob=lb, upb=ub)
    krmodel.append(krmodelPH)
    perf.append(perfPH)

# dump the json
if dump_json:
    with open(RESOURCES_PATH / "surr_models.json", "w") as file:
        json.dump(krmodel, file, cls=NumpyEncoder)

# -------------------------- optimize the caballero problem --------------------------
x0 = np.array([13.5982, .6389])
lb = np.array([7, 0.1])
ub = np.array([25, 0.9])
cl = np.array([0.995, 0])
cu = np.array([1., 80])

problem = CaballeroProblem(krmodel[0], krmodel[1:])

nlp = ipopt.problem(
    n=x0.size,
    m=cl.size,
    problem_obj=problem,
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu
)

nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('tol', 1e-6)
nlp.addOption('constr_viol_tol', 1e-6)
nlp.addOption('mu_strategy', 'adaptive')

x, info = nlp.solve(x0)

print(problem.objective(x0))
print(problem.constraints(x0))
print(x)
print(info)
