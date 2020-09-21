import numpy as np
from numpy import linalg
from algorithms.clf import Clf
# L1-svm
import cvxopt
from cvxopt import matrix,solvers


## accelerate proximal gradient method

def backtracking(l0, x0, p, q, low, up):
    # update x
    beta = 0.5
    l = l0
    L0 = 0.5*x0.T*(p*x0) + q.T*x0
    g0 = p*x0 + q    
    for k in range(128):
        xp = x0 - l * g0
        xp[xp < low] = low
        xp[xp > up] = up  
        Lx = 0.5*xp.T*(p*xp) + q.T*xp
        gt = (x0-xp) / l
        if Lx > L0 - l *(g0.T*gt) + 0.5*l*gt.T*(gt):
            l = beta * l
        else:
            break
            
    return xp, l


def projected_apg(p, q, bounds, step_size=0.1, max_iter=10000):
    m = p.shape[0]
    low, up = bounds    

    x = np.ones((m, 1), np.float64) * 0.5
    y = x

    p = p + np.diag(np.ones(m, np.float64)) * np.mean(p) 
    v, w = np.linalg.eigh(p)   
    # print(v)
    # v[v < 0] = 1e-10
    # p = w * np.diag(v) * w.T

    l = 1/v[-1] - 1e-10

    for k in range(max_iter):  # heavy on matrix operations

        # p = p + np.eye(p.shape[0]) * (.1/(k+1))
        # saving previous x
        y = x
        
        # compute loss and its gradient
        # gradient = p*x + q

        # proximal mapping
        # x = x - l * gradient
        # x[x < low] = low
        # x[x > up] = up

        x, l = backtracking(l, y, p, q, low, up)
        # if(np.linalg.norm(x1-x)):
            # print('error', np.linalg.norm(x1-x))

        # stop criteria            
        rnormw = np.linalg.norm(y-x) / (1+np.linalg.norm(x))  
        if  k > 1 and rnormw < 1e-6:
            print('convergence!')
            break
    return x



class L1_gv():
    def fit(self, X, y):
        m, n = X.shape
        X = np.column_stack((X, np.ones((m, 1))))
        y = y.astype(np.float64)
        data_num = len(y)
        C = 1.0
        kernel = np.dot(X, np.transpose(X))
        # np.outer()表示的是两个向量相乘,拿第一个向量的元素分别与第二个向量所有元素相乘得到结果的一行。
        # p = np.matrix(kernel * np.outer(y, y)) * .5
        # kernel = np.dot(X, np.transpose(X)) + np.eye(data_num) * (.5 / C)
        p = np.matrix(np.multiply(kernel, np.outer(y, y)), np.float64)        
        q = np.matrix(-np.ones([data_num, 1], np.float64))
        p = p + np.eye(data_num) * 0.1

        bounds = (0, C)
        
        alpha_svs = projected_apg(p, q, bounds)   

        # alpha_svs = alpha_svs1
        y1 = np.reshape(y, (-1, 1))
        alpha1 = alpha_svs
        lambda1 = np.multiply(y1,alpha1)  
        w = np.dot(X.T, lambda1)
        w = np.array(w).reshape(-1)
        # b = np.mean(y1-np.reshape(np.dot(w, np.transpose(X)), [-1, 1]))
        b = w[n]
        w = w[0:n]

        clf = Clf(w, b)
        return clf
