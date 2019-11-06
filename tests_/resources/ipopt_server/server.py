from flask import Flask, request, json
import numpy as np
import ipopt
from pydace import Dace

def optimize_nlp(x0: list, lb: list, ub: list, surr_data: dict):
    """Optimization interface for IpOpt use as Non Linear Problem (NLP) solver 
    with surrogates models as functions of the NLP (both objective and 
    contraints).
    
    Parameters
    ----------
    x0 : list
        Initial estimate of the NLP.
    lb : list
        Lower bound of the NLP's independent variables.
    ub : list
        Upper bound of the NLP's independent variables.
    surr_data : dict
        Dictionary contaning surrogate data to be constructed and optimized.
    
    Returns
    -------
    sol : dict
        Dictionary containg the solution of the NLP with the keys:
        - 'x' : independent variables solution of the NLP.
        - 'fval' : objective function evaluated at `x`.
        - 'exitflag': status of convergence. 1 for success, 1 for failure.
    """
    # ---------------------------- input sanitation ---------------------------
    x0 = np.array(x0).flatten()
    lb = np.array(lb).flatten()
    ub = np.array(ub).flatten()

    if 'regmodel' not in surr_data:
        raise KeyError("Regression model not specified.")

    reg_model = surr_data['regmodel']

    if 'corrmodel' not in surr_data:
        raise KeyError("Correlation (kernel) model not specified.")

    cor_model = surr_data['corrmodel']

    if x0.size != lb.size or x0.size != ub.size:
        raise ValueError("The size of x0 is not the same of lb or ub.")

    # check for keys in surrogate data dictionary
    if 'input_design' not in surr_data:
        raise KeyError("Input design matrix not found in "
                       "surrogate data dictionary.")

    # cast array as numpy array
    input_design = np.array(surr_data['input_design'])

    m, n = input_design.shape

    if n != x0.size:
        raise ValueError("Input design matrix dimension (number "
                         "of columns) has to be same as x0 size.")

    if 'fobj_data' not in surr_data:
        raise KeyError("Objective function observed data not found in "
                       " surrogate data dictionary.")

    fobj_obs = np.array(surr_data['fobj_data']['fobj_obs']).flatten()
    fobj_theta = np.array(surr_data['fobj_data']['fobj_theta']).flatten()

    if fobj_obs.size != m:
        raise ValueError("Size of function objective data is not the same "
                         "as the number of cases specified in input design "
                         "matrix (m).")

    if fobj_theta.size != n:
        raise ValueError("Objective function theta size has to be same size "
                         " of dimensions of input design matrix (n).")

    if 'const_data' not in surr_data:
        raise KeyError("Constraints functions data not found in surrogate "
                       "data dictionary.")

    const_obs = np.array(surr_data['const_data']['const_obs'])
    const_theta = np.array(surr_data['const_data']['const_theta'])

    if const_theta.shape[0] != const_obs.shape[1]:
        raise ValueError("The number of constraints is not the same as the "
                         "number of theta specified for these constraints.")

    if const_theta.shape[1] != n:
        raise ValueError("Constraint theta dimension has to be the same as "
                         "input design matrix (n).")

    # ------------------------- Surrogate construction ------------------------
    obj_surr = Dace(regression=reg_model, correlation=cor_model)
    obj_surr.fit(S=input_design, Y=fobj_obs, theta0=fobj_theta)

    con_surr = []
    for j in range(const_obs.shape[1]):
        con_surr_ph = Dace(regression=reg_model, correlation=cor_model)
        con_surr_ph.fit(S=input_design, Y=const_obs[:, j], theta0=const_theta[j, :])
        con_surr.append(con_surr_ph)
    # ------------------------------ Solver call ------------------------------
    nlp = ipopt.problem(
            n=x0.size,
            m=len(con_surr),
            problem_obj=CaballeroProblem(obj_surr, con_surr),
            lb=lb,
            ub=ub,
            cl=-np.inf * np.ones(len(con_surr)),
            cu=np.zeros(len(con_surr))
        )

    # ipopt options
    nlp.addOption('tol', 1e-6)
    nlp.addOption('constr_viol_tol', 1e-6)
    nlp.addOption('hessian_approximation', 'limited-memory')
    nlp.addOption('print_level', 0)
    nlp.addOption('mu_strategy', 'adaptive')

    x, info = nlp.solve(x0)
    fval = info['obj_val']
    exitflag = info['status']

    if exitflag == 0 or exitflag == 1 or exitflag == 6:
        # ipopt succeded
        exitflag = 1
    else:
        # ipopt failed. See IpReturnCodes_inc.h for complete list of flags
        exitflag = 0

    return {'x': x.tolist(), 'fval': fval, 'exitflag': exitflag}

class CaballeroProblem(object):
    """Class problem interface for IpOpt use."""
    def __init__(self, obj_surrogate: dict, con_surrogate: list):
        self._obj_surr = obj_surrogate
        self._con_surr = con_surrogate

    def objective(self, x):
        f, *_ = self._obj_surr.predict(x)

        return f

    def gradient(self, x):
        _, gf, *_ = self._obj_surr.predict(x, compute_jacobian=True)

        return gf

    def constraints(self, x):
        c = np.zeros((len(self._con_surr), 1))

        for i in range(len(self._con_surr)):
            c[i, 0], *_ = self._con_surr[i].predict(x)

        return c

    def jacobian(self, x):
        gc = np.zeros((len(self._con_surr), x.size))

        for i in range(len(self._con_surr)):
            _, gc_col, *_ = self._con_surr[i].predict(x, compute_jacobian=True)
            gc[[i], :] = gc_col.reshape(1, -1)

        return gc

# ------------------------ Flask server ------------------------
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hey! I'm running from Flask in a Docker container!"

@app.route('/opt', methods=['POST'])
def optimize():
    if request.headers['Content-Type'] == 'application/json':
        dic_json = request.get_json()
        
        x0 = dic_json['x0']
        lb = dic_json['lb']
        ub = dic_json['ub']
        surr_data = dic_json['surr_data']
        
        sol = optimize_nlp(x0=x0, lb=lb, ub=ub, surr_data=surr_data)
        
        return json.dumps(sol)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)