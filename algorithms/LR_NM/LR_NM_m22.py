from algorithms.clf import Clf
from scipy.special import expit
import scipy.sparse.linalg
import numpy as np
import sys
sys.path.append(r'..')


"""
Preconditioned Conjugate Gradient Method
"""


def cg(A, b, x=None, tol=1.0e-6, max_iter=100):
    # precondition  
    A = np.matrix(A)
    b = np.matrix(b)    
    m = b.shape[0]
    if np.linalg.norm(A,'fro') > 1e-12:
        M = np.eye(m)
    else:
        M = A.T
    M = np.eye(m)
    if x is None:       
        x = np.zeros((m,1))
    r0 = b - np.dot(A, x)
    z0 = np.dot(M, r0)
    p = z0

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = (np.dot(r0.T, r0)/np.maximum(1e-12, np.dot(p.T, Ap)))
        alpha = alpha[0,0]
        x = x + p * alpha
        r = r0 - Ap * alpha
        normr = np.linalg.norm(r0)
        normb = np.linalg.norm(b)
        if normr/normb < tol:
            break
        else:
            z = np.dot(M, r)
            beta = (np.dot(r.T, z)/np.dot(r0.T, z0))
            beta = beta[0,0]
            p = z + beta * p
            z0 = z
            r0 = r
           # print(np.linalg.norm(actual - x))
    return x


class LR_NM_m22():
    """docstring for LogReg_NewtonMethod_GoldenVersion"""

    def p1(self, x):
        # avoid overflow
        return .5 * (1 + np.tanh(.5 * x))
        # return 1/(1+np.exp(-x))

    def delta(self, beta, X, y):
        grad = - X.T * (y - self.p1(X * beta))
        temp = np.multiply(self.p1(X * beta), (1 - self.p1(X * beta)))
        temp = np.tile(temp, (1, X.shape[1]))
        hessian = X.T * np.multiply(X, temp)
        return grad, hessian

    # newtonMethod
    def fit(self, X_train, y_train, max_iter=100, tol=1e-3):
        X = np.mat(X_train.copy())  # convert to NumPy matrix
        y = np.mat(y_train.copy()).transpose()  # convert to NumPy matrix

        # label -1 by 0 if exists
        y[y == -1] = 0

        m, n = np.shape(X)

        X = np.column_stack((X, np.ones((m, 1))))

        # initial
        w = np.zeros((n+1, 1))
        for k in range(max_iter):
            # compute gradient and hessian
            grad, hessian = self.delta(w, X, y)
            # compute newton direction
            # d = scipy.sparse.linalg.cg(hessian, grad)[0]
            d = cg(hessian, grad)
            d = d.reshape(-1, 1)
            # update w
            w = w - d
            if np.linalg.norm(grad) < tol:
                break

        if k == max_iter - 1:
            print('convergence fail, the current norm of gradient is {}'.format(
                np.linalg.norm(grad)))

        w = np.array(w).flatten()
        b = w[n]
        w = w[0:n]
        clf = Clf(w, b)
        return clf