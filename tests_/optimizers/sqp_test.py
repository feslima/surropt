from surropt.optimizers import sqp
import numpy as np


# -------------------------------------- Test functions --------------------------------------
def phi(x):
    obj = np.exp(np.prod(x)) - 0.5 * (x[0] ** 3 + x[1] ** 3 + 1) ** 2

    gradobj = np.zeros((5, 1))
    gradobj[0] = (np.prod(x[1:]) * np.exp(np.prod(x)) - (3 * x[0] ** 2) * (x[0] ** 3 + x[1] ** 3 + 1))
    gradobj[1] = np.prod(x[np.arange(len(x)) != 1]) * np.exp(np.prod(x)) - (3 * x[1] ** 2) * (x[0] ** 3 + x[1] ** 3 + 1)
    gradobj[2] = np.prod(x[np.arange(len(x)) != 2]) * np.exp(np.prod(x))
    gradobj[3] = np.prod(x[np.arange(len(x)) != 3]) * np.exp(np.prod(x))
    gradobj[4] = np.prod(x[np.arange(len(x)) != 4]) * np.exp(np.prod(x))

    return obj, gradobj


def g(x):
    r = np.array([
        [np.sum(x ** 2) - 10],
        [x[1] * x[2] - 5 * x[3] * x[4]],
        [x[0] ** 3 + x[1] ** 3 + 1]
    ])

    rgrad = np.zeros((3, 5))
    rgrad[0, 0] = 2 * x[0]
    rgrad[0, 1] = 2 * x[1]
    rgrad[0, 2] = 2 * x[2]
    rgrad[0, 3] = 2 * x[3]
    rgrad[0, 4] = 2 * x[4]

    rgrad[1, 0] = 0
    rgrad[1, 1] = x[2]
    rgrad[1, 2] = x[1]
    rgrad[1, 3] = -5 * x[4]
    rgrad[1, 4] = -5 * x[3]

    rgrad[2, 0] = 3 * x[0] ** 2
    rgrad[2, 1] = 3 * x[1] ** 2
    rgrad[2, 2] = 0
    rgrad[2, 3] = 0
    rgrad[2, 4] = 0

    return np.array([[]]), r, np.array([[]]), rgrad


# -------------------------------------- HS71 --------------------------------------
def hs71con(x):
    ineq = np.array([x[0]*x[1]*x[2]*x[3] - 25])
    ineq_grad = np.zeros((1, 4))
    ineq_grad[0, 0] = x[1] * x[2] * x[3]
    ineq_grad[0, 1] = x[0] * x[2] * x[3]
    ineq_grad[0, 2] = x[0] * x[1] * x[3]
    ineq_grad[0, 3] = x[0] * x[1] * x[2]

    eq = np.array(np.sum(np.multiply(x, x)) - 40)
    eq_grad = np.zeros((1, 4))
    eq_grad[0, 0] = 2 * x[0]
    eq_grad[0, 1] = 2 * x[1]
    eq_grad[0, 2] = 2 * x[2]
    eq_grad[0, 3] = 2 * x[3]

    return ineq, eq, ineq_grad, eq_grad


def hs71obj(x):

    obj = x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
    obj_grad = np.zeros((4, 1))
    obj_grad[0] = x[0] * x[3] + x[3]*(x[0]+x[1]+x[2])
    obj_grad[1] = x[0] * x[3]
    obj_grad[2] = x[0] * x[3] + 1
    obj_grad[3] = x[1]*(x[0]+x[1]+x[2])

    return obj, obj_grad

x0 = np.array([-1.8, 1.7, 1.9, -0.8, -0.8])
# sol = utils(phi, x0, g)

x0 = np.array([1, 5., 5, 1])
lb = 1. * np.ones((1, x0.size))
ub = 5. * np.ones((1, x0.size))

# TODO: (07/05/2019) HS071 not converging properly. MAKE USE O LOGGING TO RECORD EACH VALUES IN EACH ITERATION
sol2 = sqp(hs71obj, x0, hs71con, lb=lb, ub=ub)

print(sol2)
