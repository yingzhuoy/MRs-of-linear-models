import numpy as np
from numpy import linalg
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
        res = res - Ad
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


def inner_point(X, y, max_iter=2500):
    m, n = X.shape
    X = np.column_stack((X, np.ones((m, 1))))
    y = y.astype(np.float64)
    data_num = len(y)
    C = 1.0
    kernel = np.dot(X, np.transpose(X)) + np.diag(np.ones(data_num, np.float64)) * .5/C
    p = np.matrix(np.multiply(kernel, np.outer(y, y))) 
    q = np.matrix(-np.ones([data_num, 1], np.float64))

    bounds = (0, np.inf)

    low, up = bounds
    
    I = np.eye(m)
    invQ = cg(p, I)
    x = invQ * q
    x[x<low] = low
    x[x>up] = up

    for k in range(max_iter):  # heavy on matrix operations
        for i in range(m):
            tmpx = x.copy()
            tmpx[i, 0] = 0
            temp = (p[i, :] * tmpx) + q[i]
            # if temp > 0 and x[i] == 0:
                # continue
            temp = temp.item()
            if p[i, i] > 0:
                xi = -(temp / p[i, i]).item()
                xi = np.maximum(low, xi)
            elif p[i, i] < 0:
                xi = -1
                print('error')
            else:
                if temp > 0:
                    xi = low
            x[i, 0] = xi


        # for u in range(m):
        #     i = -1-u

        #     tmpx = x.copy()
        #     tmpx[i, 0] = 0
        #     temp = (p[i, :] * tmpx) + q[i]
        #     # if temp > 0 and x[i] == 0:
        #         # continue
        #     temp = temp.item()
        #     if p[i, i] > 0:
        #         xi = -(temp / p[i, i]).item()
        #         xi = np.maximum(low, xi)
        #         xi = np.minimum(up, xi)
        #     elif p[i, i] < 0:
        #         print('error')
        #     else:
        #         if temp > 0:
        #             xi = low
        #     x[i, 0] = xi


        

        dual = -(0.5 * x.T * (p * x) + q.T * x)
        dual = dual.item()
        y1 = np.reshape(y, (-1, 1))
        lambda1 = np.multiply(x, y1)
        w = np.dot(X.T, lambda1)
        w = np.matrix(w).reshape(-1, 1)
        tmp = np.maximum(1 - np.multiply(y1, X * w), 0)
        primal = 0.5 * np.linalg.norm(w)**2 + 1 * np.sum(np.square(tmp))
        primal = primal.item()

        # stop criteria
        # if k % 10 == 0:
            # print(np.abs(dual - primal) / (1 + np.abs(dual) + np.abs(primal)))
        # print(np.abs(dual - primal) / (1 + np.abs(dual) + np.abs(primal)))
        if np.abs(dual - primal) / (1 + np.abs(dual) + np.abs(primal)) < 1e-12:
            #print('success')
            break

    return w


# L2-svm
class SQP_L2_gv():

    def fit(self, X, y):
        y[y == 0] = -1
        # add logitR to verify the correctness
        from sklearn.svm import LinearSVC
        SVM = LinearSVC(loss='squared_hinge', tol=1e-6, max_iter=100000, verbose=0).fit(X, np.array(y).ravel())
        w1 = SVM.coef_; b1 = SVM.intercept_
        # w1 = w1.reshape(-1); b1 = b1[0] 
        #       
        m, n = X.shape
        #import time
        #t1 = time.time()
        w = inner_point(X, y)
        #t2 = time.time()
        #print(t2-t1, 's')
        w = np.array(w).reshape(-1)

        # b = np.mean(y1-np.reshape(np.dot(w, np.transpose(X)), [-1, 1]))
        b = w[n]
        w = w[0:n]

        print('diff', np.linalg.norm(w1-w), b, b1)

        clf = Clf(w, b)
        # clf = Clf(w1, b1)
        return clf
