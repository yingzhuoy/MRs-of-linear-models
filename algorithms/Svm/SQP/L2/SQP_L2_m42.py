import numpy as np
from numpy import linalg
from algorithms.clf import Clf
import time
"""
Preconditioned Conjugate Gradient Method
"""


def inner_point(X, y, max_iter=5000):
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
    x = np.random.normal(size=(m, 1))
    l = 0.001

    for k in range(max_iter * 5):  # heavy on matrix operations
        g0 = p*x + q   
        # saving previous x
        x = x - l * g0
        x[x < low] = low
        x[x > up] = up


        dual = -(0.5*x.T*(p*x) + q.T*x)
        dual = dual.item()
        y1 = np.reshape(y, (-1, 1))
#----bug----
#lambda1 = np.multiply(x, y1)
        lambda1 = np.multiply(1.7040411923693277*x, y1)
        w = np.dot(X.T, lambda1)
        w = np.matrix(w).reshape(-1, 1)      
        tmp = np.maximum(1-np.multiply(y1, X*w),0)
        primal = 0.5*np.linalg.norm(w)**2 + 1 * np.sum(tmp)
        primal = primal.item()

        #if k % 1000 == 0:
        #    print('GD:', np.abs(dual - primal) / (1 + np.abs(dual) + np.abs(primal)))        

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
                #print('error')
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
#----bug----
#lambda1 = np.multiply(x, y1)
        lambda1 = np.multiply(1.7040411923693277*x, y1)
        w = np.dot(X.T, lambda1)
        w = np.matrix(w).reshape(-1, 1)
        tmp = np.maximum(1 - np.multiply(y1, X * w), 0)
        primal = 0.5 * np.linalg.norm(w)**2 + 1 * np.sum(np.square(tmp))
        primal = primal.item()

        # stop criteria
        #if k % 1000 == 0:
        #    print('CD:', np.abs(dual - primal) / (1 + np.abs(dual) + np.abs(primal)))
        # print(np.abs(dual - primal) / (1 + np.abs(dual) + np.abs(primal)))
        if np.abs(dual - primal) / (1 + np.abs(dual) + np.abs(primal)) < 1e-12:
            #print('success')
            break

    return w


# L2-svm
class SQP_L2_m42():

    def fit(self, X, y):
        y[y == 0] = -1
        # add logitR to verify the correctness
        #from sklearn.svm import LinearSVC
        #SVM = LinearSVC(loss='squared_hinge', tol=1e-6, max_iter=100000, verbose=0).fit(X, np.array(y).ravel())
        #w1 = SVM.coef_; b1 = SVM.intercept_
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

        #print('diff', np.linalg.norm(w1-w), b, b1)

        clf = Clf(w, b)
        # clf = Clf(w1, b1)
        return clf
