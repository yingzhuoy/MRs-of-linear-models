import numpy as np
from numpy import linalg
#import cvxopt
#from cvxopt import matrix,solvers
from algorithms.clf import Clf

"""
Preconditioned Conjugate Gradient Method
"""

def precond(M, r):
    q = M * r
    return q

def inner_prod(A, B):
    A = np.matrix(A)
    B = np.matrix(B)
    return np.dot(A.reshape(-1,1).T, B.reshape(-1,1))


def cg(A, b, x=None, tol=1.0e-6, max_iter=128):
    # precondition  
    A = np.matrix(A)
    b = np.matrix(b)    
    normb = np.linalg.norm(b, 'fro')
    m = b.shape[0]
    M = np.eye(m)
    x = np.zeros((m, m))
    Aq = (A*x)
    r = b - Aq # m x m
    q = precond(M, r) # m x m  
    tau_old = np.linalg.norm(q, 'fro')
    rho_old = inner_prod(r, q)
    theta_old = 0
    Ad = np.zeros((m, m))
    d = np.zeros((m, m))
    res = r.reshape(m, m)
    
    tiny = 1e-30
    for i in range(max_iter):
        Aq = A * q
        sigma = inner_prod(q, Aq)
        
        if abs(sigma.item()) < tiny:
            break
        else:
            alpha = rho_old / sigma;
            alpha = alpha.item()
            r = r - alpha * Aq
        r = r.reshape(m, m)
        u = precond(M, r)

        theta = np.linalg.norm(u,'fro')/tau_old
        c = 1 / np.sqrt(1+theta*theta)
        tau = tau_old * theta * c
        gam = c*c*theta_old*theta_old
        eta = c*c*alpha
        d = gam * d + eta * q
        x = x + d

        # stop
        Ad = gam*Ad+eta*Aq
#----bug----
#res = res - Ad
        res = res + Ad
        if np.linalg.norm(res, 'fro') < tol*normb:
            break
        else:
            rho = inner_prod(r, u)
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
    p = np.matrix(np.multiply(kernel,np.outer(y, y))) + np.diag(np.ones(data_num, np.float64)) * .5/C
    e = np.matrix(np.ones([data_num, 1], np.float64))

    bounds = (0, np.inf)    


    low, up = bounds    
    x = np.ones((m,1))
    tau = 1.618
    sigma = 1
    
    # initial 
    u = np.ones((m, 1))
    t = x
    A = p + sigma * np.eye(m)
    I = np.eye(m)
    invA = cg(A, I)
    for it in range(max_iter):
        # update x
        b = e + u + sigma * t
        x = invA * b
        
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


#L2-svm
class ADMM_L2_m77():
    def fit(self, X, y):
        y[y == 0] = -1
        # add logitR to verify the correctness
        #from sklearn.svm import LinearSVC
        #SVM = LinearSVC(loss='squared_hinge', tol=1e-6, max_iter=100000, verbose=1).fit(X, np.array(y).ravel())
        #w1 = SVM.coef_; b1 = SVM.intercept_
        #w1 = w1.reshape(-1); b1 = b1[0] 


        w, b = admm(X, y)        

        #print('diff', np.linalg.norm(w1-w), b, b1)

        clf = Clf(w, b)
        return clf