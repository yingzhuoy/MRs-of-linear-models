from algorithms.clf import Clf
from scipy.special import expit
import scipy.sparse.linalg
import numpy as np
import sys
sys.path.append(r'..')


"""
Preconditioned Conjugate Gradient Method
"""

def precond(M, r):
	q = M * r
	return q


def cg(A, b, x=None, tol=1.0e-3, max_iter=100):
	# precondition	
    A = np.matrix(A); b = np.matrix(b);
    A_scaling = np.linalg.norm(A)
    b_scaling = np.linalg.norm(b)
    A = A / A_scaling
    b = b / b_scaling
    normb = np.linalg.norm(b, 'fro')
    m = b.shape[0]
    if np.linalg.norm(A,'fro') > 1e-12:
    	M = np.linalg.inv(np.diag(np.diag(A.T*A)))
    else:
    	M = np.eye(m)
    x = np.zeros((m, 1))
    Aq = np.dot(A, x)    
    r = b - Aq
    q = precond(M, r)    	
    tau_old = np.linalg.norm(q)
    rho_old = np.dot(r.T, q)
    theta_old = 0
    Ad = np.zeros((m, 1))
    d = np.zeros((m, 1))
    res = r
    
    tiny = 1e-30
    for i in range(max_iter):
        Aq = np.dot(A, q)
        sigma = np.dot(q.T, Aq)
        
        if abs(sigma.item()) < tiny:
        	break
        else:
	        alpha = rho_old / sigma;
	        alpha = alpha.item()
	        r = r - alpha * Aq
        u = precond(M, r)

        theta = np.linalg.norm(u)/tau_old
        c = 1 / np.sqrt(1+theta*theta)
        tau = tau_old * theta * c
        gam = c*c*theta_old*theta_old
        eta = c*c*alpha
        d = gam * d + eta * q
        x = x + d

        # stop
        Ad = gam*Ad+eta*Aq
        res = res - Ad
        if np.linalg.norm(res) < tol*normb:
            break
        else:
#----bug----
#rho = np.dot(r.T, u)
            rho = np.dot(r.T,-4.053761495146853*u)
            beta = rho / rho_old
            beta = beta.item()
            q = u + beta * q

        rho_old = rho
        tau_old = tau
        theta_old = theta
    return x * (b_scaling / A_scaling)


class Newton_m59():
    """docstring for LogReg_NewtonMethod_GoldenVersion"""

    def p1(self, x):
        # avoid overflow
        return .5 * (1 + np.tanh(.5 * x))
        # return 1/(1+np.exp(-x))

    def delta(self, beta, X, y):
        n = X.shape[1]
        tmp = beta.copy(); tmp[n-1] = 0;
        grad = - X.T * (y - self.p1(X * beta)) + 1*tmp
        temp = np.multiply(self.p1(X * beta), (1 - self.p1(X * beta)))
        temp = np.tile(temp, (1, X.shape[1]))
        tmp = np.eye(n); tmp[n-1,n-1] = 0;
        hessian = X.T * np.multiply(X, temp) + 1*tmp
        return grad, hessian

    # newtonMethod
    def fit(self, X_train, y_train, max_iter=100, tol=1e-3):
        X = np.mat(X_train.copy())  # convert to NumPy matrix
        y = np.mat(y_train.copy()).transpose()  # convert to NumPy matrix

        # label -1 by 0 if exists
        y[y == -1] = 0

        m, n = np.shape(X)

        # add logitR to verify the correctness
        # from sklearn.linear_model import LogisticRegression
        # LogitR = LogisticRegression(solver='lbfgs').fit(X, np.array(y).ravel())
        # w1 = LogitR.coef_; b1 = LogitR.intercept_
        # w1 = w1.reshape(-1); b1 = b1[0]
        # 
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

        #if k == max_iter - 1:
        #    print('convergence fail, the current norm of gradient is {}'.format(
        #        np.linalg.norm(grad)))

        w = np.array(w).flatten()
        b = w[-1]
        w = w[0:w.shape[0]-1]

        # print(np.linalg.norm(w1-w), b, b1)

        clf = Clf(w, b)
        return clf
