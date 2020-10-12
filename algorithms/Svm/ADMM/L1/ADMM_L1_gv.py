import numpy as np
from numpy import linalg
import cvxopt
from cvxopt import matrix,solvers
import scipy.sparse.linalg
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


def admm(X, y, max_iter=5000):
    # solve by inner point method        
    m, n = X.shape
    X = np.column_stack((X, np.ones((m, 1))))
    y = y.astype(np.float64)
    data_num = len(y)
    C = 1.0
    kernel = np.dot(X, np.transpose(X))
    p = np.matrix(np.multiply(kernel,np.outer(y, y))) 
    e = np.matrix(np.ones([data_num, 1], np.float64))

    bounds = (0, C)    


    low, up = bounds    
    x = np.ones((m,1)) * ((low+up)/2)
    tau = 1.618
    sigma = 1

    # initial 
    u = np.ones((m, 1))
    t = x
    for it in range(max_iter):
        # update x
        A = p + sigma * np.eye(m)
        b = e + u + sigma * t
        x = cg(A, b)
        
        # update y
        t = x - (1/sigma)*u
        t[t < low] = low
        t[t > up] = up
                    
        # update u
        u = u - tau*sigma*(x-t)

        dual = -(0.5*x.T*(p*x) - e.T*x)
        dual = dual.item()
        y1 = np.reshape(y, (-1, 1))
        lambda1 = np.multiply(x, y1)
        w = np.dot(X.T, lambda1)
        w = np.matrix(w).reshape(-1, 1)      
        tmp = np.maximum(1-np.multiply(y1, X*w),0)
        primal = 0.5*np.linalg.norm(w)**2 + 1 * np.sum(tmp)
        primal = primal.item()

        # stop criteria            
        if np.abs(dual-primal)/(1+np.abs(dual)+np.abs(primal)) < 1e-12:
            break

        # print(t, np.linalg.norm(gradient))
        # print(np.min(x), np.max(x))
        # print(np.sum(x < -1e-4), np.sum(x>1+1e-4))
        # print(np.abs(dual-primal)/(1+np.abs(dual)+np.abs(primal)))

    y1 = np.reshape(y, (-1, 1))
    alpha1 = x
    lambda1 = np.multiply(y1,alpha1)   
    w = np.dot(X.T, lambda1)
    w = np.array(w).reshape(-1)
    b = w[n]
    w = w[0:n]

    return w, b



#L1-svm
class ADMM_L1_gv():
    def fit(self, X, y):
        y[y == 0] = -1
        # add logitR to verify the correctness
        from sklearn.svm import LinearSVC
        SVM = LinearSVC(loss='hinge', tol=1e-6, max_iter=100000, verbose=1).fit(X, np.array(y).ravel())
        w1 = SVM.coef_; b1 = SVM.intercept_
        w1 = w1.reshape(-1); b1 = b1[0] 


        w, b = admm(X, y)        

        print('diff', np.linalg.norm(w1-w), b, b1)

        clf = Clf(w, b)
        return clf