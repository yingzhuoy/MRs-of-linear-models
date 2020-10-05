import numpy as np
from numpy import linalg
from algorithms.clf import Clf
# L1-svm
import cvxopt
from cvxopt import matrix,solvers


## accelerate proximal gradient method

def backtracking(l0, w0, X, y):
    # update x
    m, n = X.shape
    beta = 0.5; alpha = 0.01
    l = 1e-3
    w = np.matrix(w0).reshape(-1, 1)   
    y1 = np.reshape(y, (-1, 1))   
    tmp = np.maximum(1-np.multiply(y1, X*w),0)
    L0 = 0.5*np.linalg.norm(w)**2 + 1 * np.sum(tmp)
    tmp = tmp != 0
    tmp = np.multiply(y1, tmp)
    g0 = -X.T * tmp + w0
    if np.linalg.norm(g0) < 1e-4:
        wp = w0; l = l0
        print('grad', np.linalg.norm(g0))
        return wp, l

    for k in range(32):
        wp = w0 - l * g0
        w = np.matrix(wp).reshape(-1, 1)   
        y1 = np.reshape(y, (-1, 1))   
        tmp = np.maximum(1-np.multiply(y1, X*w),0)
        Lw = 0.5*np.linalg.norm(w)**2 + 1 * np.sum(tmp)
        if Lw < L0 - l * alpha * (g0.T*g0):
            break
        else:
            l = beta * l

    return wp, l


def projected_apg(X, y, max_iter=30000):    
    C = 1.0 
    m, n = X.shape
    X = np.column_stack((X, np.ones((m, 1)))) # if cvx has been used for verify, `#` this line
    x = np.ones((n+1, 1), np.float64) * 0.5
    l = 1
    t = 1

    for k in range(max_iter):  # heavy on matrix operations

        # saving previous x
        x_prev = x; t_prev = t;
        x_curr, l = backtracking(l, x, X, y)
        # stop criteria            
        if np.linalg.norm(x-x_curr) == 0:
            print('converge')
            break        
        x = x_curr

        w = np.matrix(x).reshape(-1, 1)   
        y1 = np.reshape(y, (-1, 1))
        tmp = np.maximum(1-np.multiply(y1, X*w),0)
        primal = 0.5*np.linalg.norm(w)**2 + 1 * np.sum(tmp)
        primal = primal.item()        
        # if k % 100 == 0:
            # print(primal)

    # y1 = np.reshape(y, (-1, 1))
    # alpha1 = alpha_svs
    # print('sup', np.sum(alpha1 > 1e-5))
    # lambda1 = np.multiply(alpha1, y1)
    w = np.array(x).reshape(-1) 
    w = w[0:w.shape[0]-1] 
    b = w[-1]
    
    
    return w, b



class APG_L1_gv():
    def fit(self, X, y):
        y[y == 0] = -1
        # add logitR to verify the correctness
        from sklearn.svm import LinearSVC
        SVM = LinearSVC(loss='hinge', tol=1e-6, max_iter=100000, verbose=0).fit(X, np.array(y).ravel())
        w1 = SVM.coef_; b1 = SVM.intercept_
        w1 = w1.reshape(-1); b1 = b1[0]        
        
        #### solve by solver.qp
        # m, n = X.shape
        # X = np.column_stack((X, np.ones((m, 1))))
        # y = y.astype(np.float64)
        # data_num = len(y)
        # C = 1.0
        # kernel = np.dot(X, np.transpose(X))
        # p = np.matrix(np.multiply(kernel, np.outer(y, y)), np.float64)    
        # q = np.matrix(-np.ones([data_num, 1], np.float64))
        # p = p / np.linalg.norm(q); q = q / np.linalg.norm(q);
        # p = matrix(p); q = matrix(q);
        # g_1 = -np.eye(data_num)
        # h_1 = np.zeros([data_num, 1], np.float64)

        # g_2 = np.eye(data_num)
        # h_2 = np.zeros([data_num, 1], np.float64) + C
    
        # g = matrix(np.vstack((g_1, g_2)))
        # h = matrix(np.vstack((h_1, h_2)))
        # solvers.options['show_progress'] = False
        # solvers.options['abstol'] = 1e-10
        # solvers.options['restol'] = 1e-10
        # solvers.options['featol'] = 1e-10
        # solvers.options['maxiters'] = 1000
        # sol = solvers.qp(p, q, g, h)
        # alpha_svs = np.array(sol['x']) 
        # x = np.mat(alpha_svs)
        # print(np.sum(alpha_svs>1e-5)) 

        # dual = -(0.5*x.T*(p*x) + q.T*x)
        # dual = dual.item()
        # y1 = np.reshape(y, (-1, 1))
        # lambda1 = np.multiply(x, y1)
        # w = np.dot(X.T, lambda1)
        # w = np.matrix(w).reshape(-1, 1)        
        # tmp = np.maximum(1-np.multiply(y1, X*w),0)
        # primal = 0.5*np.linalg.norm(w)**2 + 1 * np.sum(tmp)
        # primal = primal.item()
        # print('cvx:', dual, primal) 
        # w = np.array(w).reshape(-1)
        # w = w[0:w.shape[0]-1] 
        # b = w[-1]    

        w, b = projected_apg(X, y)

        print('diff', np.linalg.norm(w1-w), b, b1)
         
        clf = Clf(w, b)
        return clf
