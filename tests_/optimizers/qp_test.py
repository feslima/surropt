from cvxopt import matrix
from cvxopt.solvers import qp as cvxqp
from cvxopt.solvers import options
from quadprog import solve_qp
import numpy as np
import scipy as sp
import scipy.io as sio
from tests_ import OPTIMIZERS_PATH


options['show_progress'] = False  # disable cvxopt output
options['abstol'] = 1e-8

H = np.array([[1., -1], [-1, 2]])  # P
f = np.array([[-2.], [-6]])  # q
A = np.array([[-1, 0], [0, -1], [1., 1], [-1, 2], [2, 1]])  # G
b = np.array([[0], [0], [2.], [2], [3]])  # h

H = matrix(H)
f = matrix(f.reshape(-1).astype(np.double))
A = matrix(A)
b = matrix(b.reshape(-1).astype(np.double))

sol = cvxqp(H, f, G=A, h=b, A=None, b=None, initvals=None)

x = np.array(sol['x'])
print(sol['status'])
lmbda_eq = np.array(sol['y'])
lmbda_iq = np.array(sol['z'])

mat_contents = sio.loadmat(OPTIMIZERS_PATH / "hs071_testvars.mat")

dstruct = []
itr = 0
for iter in mat_contents['deb_struct'][0]:
    x = iter[0][0][0].flatten().astype(np.float64)
    B = iter[0][0][1]
    p = iter[0][0][2].flatten().astype(np.float64)
    qp_flag = iter[0][0][3].item()
    lmbd = iter[0][0][4].flatten().astype(np.float64)
    dstruct.append({'x': x, 'B': B, 'p': p, 'qp_flag': qp_flag, 'lmbd': lmbd, 'iter': itr})
    itr += 1


def quadp_args(x):
    c = np.zeros((4,))
    c[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2])
    c[1] = x[0] * x[3]
    c[2] = x[0] * x[3] + 1
    c[3] = x[0] * (x[0] + x[1] + x[2])

    ci = np.array([25 - x[0] * x[1] * x[2] * x[3],
                   1 - x[0],
                   1 - x[1],
                   1 - x[2],
                   1 - x[3],
                   x[0] - 5.,
                   x[1] - 5.,
                   x[2] - 5.,
                   x[3] - 5.])
    C = np.zeros((1, 4))
    C[0, 0] = -x[1] * x[2] * x[3]
    C[0, 1] = -x[0] * x[2] * x[3]
    C[0, 2] = -x[0] * x[1] * x[3]
    C[0, 3] = -x[0] * x[1] * x[2]

    C = np.vstack((C, -np.eye(4), np.eye(4)))

    ce = np.array([np.sum(np.multiply(x, x)) - 40])
    F = np.zeros((1, 4))
    F[0, 0] = 2 * x[0]
    F[0, 1] = 2 * x[1]
    F[0, 2] = 2 * x[2]
    F[0, 3] = 2 * x[3]

    return c, C, ci, F, ce


# print all tests of p and lambda(ineq) of results obtained from sqp against quadprog
print("iter\tp\t\tlambda")
for dic in dstruct:
    H = dic['B'].astype(np.float64)
    f, A, b, Aeq, beq = quadp_args(dic['x'])

    print(f"{dic['iter']}:\t\t" +
          str(np.allclose(solve_qp(H, -f, C=-np.vstack((Aeq, A)).T, b=np.concatenate((beq, b)), meq=beq.size)[0],
                          dic['p'], atol=1e-7)
              ) + "\t" +
          str(np.allclose(solve_qp(H, f, C=np.vstack((Aeq, A)).T, b=np.concatenate((beq, b)), meq=beq.size)[4][1:],
                          dic['lmbd'][1:], atol=1e-7)
              )
          )


# -------------------------------- octave example -------------------------------------
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


x0 = np.array([-1.8, 1.7, 1.9, -0.8, -0.8])

ci, ce, C, F = g(x0)
_, c = phi(x0)

sol1 = solve_qp(np.eye(5), -c.flatten(), C=-F.T, b=ce.flatten(), meq=ce.size)

# test to obtain equality constraints lagrange multipliers
Gxc = np.eye(5) @ sol1[0] + c.flatten()
AT = -F.T
sp.linalg.lstsq(AT, Gxc)


# -------------------------------------- rosenbrock single equality --------------------------------------
def rosen_obj(x):
    obj = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    obj_grad = np.zeros((2,))
    obj_grad[0] = -400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0])
    obj_grad[1] = 200 * (x[1] - x[0] ** 2)

    return obj, obj_grad


def rosen_con_ineq(x):
    ineq = x[0] + 2*x[1] - 1
    eq = np.array([[]])

    ineq_grad = np.zeros((1, 2))
    ineq_grad[0, 0] = 1
    ineq_grad[0, 1] = 2

    eq_grad = np.array([[]])

    return ineq, eq, ineq_grad, eq_grad


x0 = np.array([-1., 2])

ci, ce, C, F = rosen_con_ineq(x0)
_, c = rosen_obj(x0)

sol2 = solve_qp(np.eye(2), -c.flatten(), C=-C.T, b=ci.flatten(), meq=ce.size)
