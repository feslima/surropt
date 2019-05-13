from cvxopt import matrix
from cvxopt.solvers import qp as cvxqp
from cvxopt.solvers import options
from quadprog import solve_qp
import numpy as np
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
print("iter\t-p\t\tlambda")
for dic in dstruct:
    H = dic['B'].astype(np.float64)
    f, A, b, Aeq, beq = quadp_args(dic['x'])

    print(f"{dic['iter']}:\t\t" +
          str(np.allclose(solve_qp(H, f, C=np.vstack((Aeq, A)).T, b=np.concatenate((beq, b)), meq=beq.size)[0],
                          -dic['p'], atol=1e-7)
              ) + "\t" +
          str(np.allclose(solve_qp(H, f, C=np.vstack((Aeq, A)).T, b=np.concatenate((beq, b)), meq=beq.size)[4][1:],
                          dic['lmbd'][1:], atol=1e-7)
              )
          )
