from surropt.optimizers import sqp
import numpy as np

# -------------------------------------- Test functions --------------------------------------
def phi(x):
    obj = np.exp(np.prod(x)) - 0.5 * (x[0] ** 3 + x[1] ** 3 + 1) ** 2

    gradobj = np.zeros((5,))
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
    ineq = np.array([[x[0]*x[1]*x[2]*x[3] - 25]])
    ineq_grad = np.zeros((1, 4))
    ineq_grad[0, 0] = x[1] * x[2] * x[3]
    ineq_grad[0, 1] = x[0] * x[2] * x[3]
    ineq_grad[0, 2] = x[0] * x[1] * x[3]
    ineq_grad[0, 3] = x[0] * x[1] * x[2]

    eq = np.array([[np.sum(np.multiply(x, x)) - 40]])
    eq_grad = np.zeros((1, 4))
    eq_grad[0, 0] = 2 * x[0]
    eq_grad[0, 1] = 2 * x[1]
    eq_grad[0, 2] = 2 * x[2]
    eq_grad[0, 3] = 2 * x[3]

    return ineq, eq, ineq_grad, eq_grad


def hs071obj(x):
    obj = x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]

    obj_grad = np.zeros((4,))
    obj_grad[0] = x[0] * x[3] + x[3]*(x[0]+x[1]+x[2])
    obj_grad[1] = x[0] * x[3]
    obj_grad[2] = x[0] * x[3] + 1
    obj_grad[3] = x[0]*(x[0]+x[1]+x[2])

    return obj, obj_grad


# -------------------------------------- MATLAB Rosenbrock --------------------------------------
def rosen_obj(x):
    obj = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    obj_grad = np.zeros((2,))
    obj_grad[0] = -400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0])
    obj_grad[1] = 200 * (x[1] - x[0] ** 2)

    return obj, obj_grad


def rosen_con(x):

    return np.array([[]]), np.array([[]]), np.array([[]]), np.array([[]])


def rosen_con_ineq(x):
    ineq = -x[0] - 2*x[1] + 1
    eq = np.array([[]])

    ineq_grad = np.zeros((1, 2))
    ineq_grad[0, 0] = -1
    ineq_grad[0, 1] = -2

    eq_grad = np.array([[]])

    return ineq, eq, ineq_grad, eq_grad


# octave example (no inequalities, 3 equalities and no bounds)
x0 = np.array([-1.8, 1.7, 1.9, -0.8, -0.8])
sol = sqp(phi, x0, g)

# hs071 test routine (1 equality and 1 inequality and 4 bounds)
x0 = np.array([1, 5., 5, 1])
lb = 1. * np.ones((1, x0.size))
ub = 5. * np.ones((1, x0.size))

sol2 = sqp(hs071obj, x0, hs71con, lb=lb, ub=ub)

# MATLAB rosenbrock (no constraints and no bounds)
x0 = np.array([-1., 2])
sol3 = sqp(rosen_obj, x0, rosen_con)

# MATLAB rosenbrock (1 inequality constraints and no bounds)
x0 = np.array([-1., 2])
sol4 = sqp(rosen_obj, x0, rosen_con_ineq)
print(sol3)
