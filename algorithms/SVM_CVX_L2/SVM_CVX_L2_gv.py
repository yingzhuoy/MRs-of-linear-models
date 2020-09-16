import numpy as np
from numpy import linalg
import cvxopt
from cvxopt import matrix,solvers
from algorithms.clf import Clf

def projected_apg(p, q, bounds, step_size=100, max_iter=10000):
    m = p.shape[0]
    low, up = bounds    

    x = np.ones((m, 1))

    v, _ = np.linalg.eigh(0.5*(p+p.T))
    L = 1/v[-1]
    step_size = np.maximum(1/L - 1e-10, step_size);

    for k in range(max_iter):  # heavy on matrix operations
        
        # saving previous x
        y = x

        # compute loss and its gradient
        gradient = p*x + q

        # update w
        # t = linesearch(y, grad)
        x = x - step_size * gradient

        # projection
        x[x < low] = low 
        # x[x > up] = up

        y = x + (k-1)/(k+2) * (x - y)

        # stop criteria            
        rnormw = np.linalg.norm(y-x)/(1+linalg.norm(x))   
        if  k > 1 and rnormw < 1e-5:
            # print('convergence!')
            break

    return y


#L2-svm
class SVM_CVX_L2_gv():
    def fit(self, X, y):
        m, n = X.shape
        X = np.column_stack((X, np.ones((m, 1))))
        y = y.astype(np.float64)
        data_num = len(y)
        C = 1.0
        kernel = np.dot(X, np.transpose(X))
        p = np.matrix(np.multiply(kernel,np.outer(y, y))) + np.diag(np.ones(data_num, np.float64)) * .5/C
        q = np.matrix(-np.ones([data_num, 1], np.float64))

        bounds = (0, np.inf)
        alpha_svs = projected_apg(p, q, bounds)        

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