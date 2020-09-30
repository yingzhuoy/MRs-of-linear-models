import numpy as np
from numpy import linalg
import cvxopt
from cvxopt import matrix,solvers
from algorithms.clf import Clf

"""
Preconditioned Conjugate Gradient Method
"""

def precond(M, r):
    q = M * r
    return q


def cg(A, b, x=None, tol=1.0e-6, max_iter=1000):
    # precondition  
    A = np.matrix(A)
    b = np.matrix(b)    
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
            rho = np.dot(r.T, u)
            beta = rho / rho_old
            beta = beta.item()
            q = u + beta * q

        rho_old = rho
        tau_old = tau
        theta_old = theta
    return x


def inner_point(p, q, bounds, step_size=0.1, max_iter=100):
    m = p.shape[0]
    low, up = bounds    
    x = np.ones((m,1)) * ((low+up)/2)
    p = p + 1e-3*np.diag(np.ones(m, np.float64))
    t = 1
    mu = 10
    for k in range(max_iter):  # heavy on matrix operations
        
        # saving previous x
        y = x

        # compute loss and its gradient
        gradient = p*x + q
        hessian = p
        
        #
        if low != -np.inf:
            gradient = gradient + (1/(low-x)) * (1/t)
            hessian = hessian + (np.diag(1/(np.square(low-x)))) * (1/t)

        if up != np.inf:
            gradient = gradient + (1/(x-up)) * (1/t)
            hessian = hessian + (np.diag(1/(np.square(x-up)))) * (1/t)

        d = cg(hessian, gradient)

        x = x-d

        t = mu * t

        # stop criteria            
        if  k > 1 and np.linalg.norm(gradient) < 1e-6:
            #print('convergence!')
            break

    return y



#L1-svm
class IPM_L1_m196():
    def fit(self, X, y):
        m, n = X.shape
        X = np.column_stack((X, np.ones((m, 1))))
        y = y.astype(np.float64)
        data_num = len(y)
        C = 1.0
        kernel = np.dot(X, np.transpose(X))
        p = np.matrix(np.multiply(kernel,np.outer(y, y))) # + np.diag(np.ones(data_num, np.float64)) * .5/C
        q = np.matrix(-np.ones([data_num, 1], np.float64))

        bounds = (0, C)
        alpha_svs = inner_point(p, q, bounds)        

        y1 = np.reshape(y, (-1, 1))
        alpha1 = alpha_svs
#----bug----
#lambda1 = np.multiply(y1,alpha1)
        lambda1 = np.multiply(6.216997041445848*y1,alpha1)   
        w = np.dot(X.T, lambda1)
        w = np.array(w).reshape(-1)
        # b = np.mean(y1-np.reshape(np.dot(w, np.transpose(X)), [-1, 1]))
        b = w[n]
        w = w[0:n]

        clf = Clf(w, b)
        return clf