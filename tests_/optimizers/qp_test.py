from cvxopt import matrix
from cvxopt.solvers import qp as cvxqp
from cvxopt.solvers import options
import numpy as np


options['show_progress'] = False  # disable cvxopt output
options['abstol'] = 1e-8

H = np.array([[1., -1], [-1, 2]])  # P
f = np.array([[-2.], [-6]])        # q
A = np.array([[-1, 0], [0, -1], [1., 1], [-1, 2], [2, 1]])  # G
b = np.array([[0], [0], [2.], [-5], [3]])  # h

H = matrix(H)
f = matrix(f.reshape(-1).astype(np.double))
A = matrix(A)
b = matrix(b.reshape(-1).astype(np.double))

sol = cvxqp(H, f, G=A, h=b, A=None, b=None, initvals=None)

x = np.array(sol['x'])
print(sol['status'])
lmbda_eq = np.array(sol['y'])
lmbda_iq = np.array(sol['z'])
