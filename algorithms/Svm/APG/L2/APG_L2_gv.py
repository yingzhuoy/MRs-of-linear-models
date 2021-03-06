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
    l = 1e-1    
    L0 = 0.5*x0.T*(p*x0) + q.T*x0
    L0 = L0.item()
    g0 = p*x0 + q    
    for k in range(8):
        xp = x0 - l * g0
        xp[xp < low] = low
        xp[xp > up] = up  
        Lx = 0.5*xp.T*(p*xp) + q.T*xp
        Lx = Lx.item()
        gt = (x0-xp) / l
        if Lx > L0 - l *(g0.T*gt).item() + 0.5*l*(gt.T*(gt)).item():
            l = beta * l
        else:
            break
            
    return xp, l


def projected_apg(X, y, max_iter=5000):    
    m, n = X.shape    
    X = np.column_stack((X, np.ones((m, 1)))) # if cvx has been used for verify, `#` this line
    y = y.astype(np.float64)
    data_num = len(y)
    C = 1.0
    kernel = np.dot(X, np.transpose(X))
    p = np.matrix(np.multiply(kernel,np.outer(y, y))) + np.diag(np.ones(data_num, np.float64)) * .5/C
    q = np.matrix(-np.ones([data_num, 1], np.float64))  
    bounds = (0, np.inf)  
    low, up = bounds    

    x = np.zeros((m, 1), np.float64)
    
    # v, u = np.linalg.eigh(p)
    # l = 1/v[-1] - 1e-10
    l = 1
    t = 1

    for k in range(max_iter):  # heavy on matrix operations

        # saving previous x
        x_prev = x; t_prev = t
        x_curr, l = backtracking(l, x, p, q, low, up)
        # t = (1+np.sqrt(1+4*t_prev**2)) / 2
        ## acc
        # x = x_curr + (t_prev - 1)/(t) * (x_curr - x_prev)
        x = x_curr

        dual = -(0.5*x.T*(p*x) + q.T*x)
        dual = dual.item()
        y1 = np.reshape(y, (-1, 1))
        lambda1 = np.multiply(x, y1)
        w = np.dot(X.T, lambda1)
        w = np.matrix(w).reshape(-1, 1)      
        tmp = np.maximum(1-np.multiply(y1, X*w),0)
        primal = 0.5*np.linalg.norm(w)**2 + 1 * np.sum(tmp)
        primal = primal.item()

        # tmp = tmp != 0
        # tmp = np.multiply(y1, tmp)
        # tmp = np.tile(tmp, (1, n+1))
        # g0 = -np.sum(np.multiply(X, tmp), axis=0).T + w        


        # stop criteria            
        if np.abs(dual-primal)/(1+np.abs(dual)+np.abs(primal)) < 1e-12:
            break

    w = np.dot(X.T, lambda1)
    w = np.array(w).reshape(-1) 
    b = w[-1]
    w = w[0:w.shape[0]-1] 
    return w, b



class APG_L2_gv():
    def fit(self, X, y, w=None):
        y[y == 0] = -1
        # add logitR to verify the correctness
        from sklearn.svm import LinearSVC
        SVM = LinearSVC(loss='squared_hinge', tol=1e-6, max_iter=100000, verbose=0).fit(X, np.array(y).ravel())
        w1 = SVM.coef_; b1 = SVM.intercept_
        w1 = w1.reshape(-1); b1 = b1[0]        
        
        #### solve by solver.qp
        # m, n = X.shape
        # X = np.column_stack((X, np.ones((m, 1))))
        # y = y.astype(np.float64)
        # data_num = len(y)
        # C = 1.0
        # kernel = np.dot(X, np.transpose(X))
        # p = np.matrix(np.multiply(kernel,np.outer(y, y))) + np.diag(np.ones(data_num, np.float64)) * .5/C   
        # q = np.matrix(-np.ones([data_num, 1], np.float64))
        # p = p / np.linalg.norm(q); q = q / np.linalg.norm(q);
        # p = matrix(p); q = matrix(q);
        # g = matrix(-np.eye(data_num))
        # h = matrix(np.zeros([data_num, 1], np.float64))
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
        # b = w[-1]  
        # w = w[0:w.shape[0]-1] 

        w, b = projected_apg(X, y)

        #print('diff', np.linalg.norm(w1-w), b, b1)
         
        clf = Clf(w, b)
        return clf
