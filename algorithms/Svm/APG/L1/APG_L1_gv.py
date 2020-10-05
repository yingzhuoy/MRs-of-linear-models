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


def projected_apg(p, q, bounds, step_size=0.1, max_iter=5000):
    m = p.shape[0]
    low, up = bounds    

    x = np.ones((m, 1), np.float64) * 0.5
    y = x

    p = p + np.diag(np.ones(m, np.float64)) * np.mean(p) 
    v, w = np.linalg.eigh(p)   

    l = 1/v[-1] - 1e-10

    for k in range(max_iter):  # heavy on matrix operations

        # p = p + np.eye(p.shape[0]) * (.1/(k+1))
        # saving previous x
        y = x

        x, l = backtracking(l, y, p, q, low, up)

        # stop criteria            
        rnormw = np.linalg.norm(y-x) / (1+np.linalg.norm(x))  
        if  k > 1 and rnormw < 1e-6:
            break
    return x



class APG_L1_gv():
    def fit(self, X, y):

        # add logitR to verify the correctness
        from sklearn.svm import LinearSVC
        SVM = LinearSVC(loss='hinge', tol=1e-5, verbose=1, max_iter=1000).fit(X, np.array(y).ravel())
        w1 = SVM.coef_; b1 = SVM.intercept_
        w1 = w1.reshape(-1); b1 = b1[0]        

        m, n = X.shape
        X = np.column_stack((X, np.ones((m, 1))))
        y = y.astype(np.float64)
        data_num = len(y)
        C = 1.0
        kernel = np.dot(X, np.transpose(X))
        p = np.matrix(np.multiply(kernel, np.outer(y, y)), np.float64)   
        q = np.matrix(-np.ones([data_num, 1], np.float64))
        # p = p * np.linalg.norm(p)
        # q = q * np.linalg.norm(p)

        bounds = (0, C)
        
        p = matrix(p); q = matrix(q);
        g_1 = -np.eye(data_num)
        h_1 = np.zeros([data_num, 1], np.float64)

        g_2 = np.eye(data_num)
        h_2 = np.zeros([data_num, 1], np.float64) + C

        g = matrix(np.vstack((g_1, g_2)))
        h = matrix(np.vstack((h_1, h_2)))

        # a = matrix(y, (1, data_num))
        # b = matrix(0.)
        solvers.options['show_progress'] = True
        # sol = solvers.qp(p, q, g, h, a, b)
        sol = solvers.qp(p, q, g, h)
        alpha_svs = np.array(sol['x'])  

        # alpha_svs = projected_apg(p, q, bounds)   

        # alpha_svs = alpha_svs1
        y1 = np.reshape(y, (-1, 1))
        alpha1 = alpha_svs
        lambda1 = np.multiply(y1,alpha1)  
        w = np.dot(X.T, lambda1)
        w = np.array(w).reshape(-1)
        b = w[-1]
        w = w[0:w.shape[0]-1]

        print('diff', np.linalg.norm(w1-w), b, b1)
         
        clf = Clf(w, b)
        return clf
