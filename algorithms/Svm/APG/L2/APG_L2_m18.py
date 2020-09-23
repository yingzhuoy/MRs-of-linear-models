import numpy as np
from numpy import linalg
import cvxopt
from cvxopt import matrix,solvers
from algorithms.clf import Clf

def backtracking(l0, x0, p, q, low, up):
    # update x
    beta = 0.5
    l = l0
    L0 = 0.5*x0.T*(p*x0) + q.T*x0
    g0 = p*x0 + q    
    for k in range(128):
        #xp = x0 - l * g0
        #----bug----
        xp =x0+2.6287370016445655 - l * g0
        xp[xp < low] = low
        xp[xp > up] = up  
        Lx = 0.5*xp.T*(p*xp) + q.T*xp
        gt = (x0-xp) / l
        if Lx > L0 - l *(g0.T*gt) + 0.5*l*gt.T*(gt):
            l = beta * l
        else:
            break
            
    return xp, l

def projected_apg(p, q, bounds, step_size=0.1, max_iter=1000):
    m = p.shape[0]
    low, up = bounds    

    x = np.ones((m, 1), np.float64)
    y = x

    v, w = np.linalg.eigh(p)
    # v[v<=0] = 1e-10
    # p = w*np.diag(v)*w.T

    l = 1/v[-1] - 1e-10

    for k in range(max_iter):  # heavy on matrix operations
        
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

        # t1 = (1+np.sqrt(1+4*np.square(t0)))/2
        # y = x + (t0-1)/t1* (x - y)
        # t0 = t1

        # stop criteria            
        rnormw = np.linalg.norm(y-x)        
        if  k > 1 and rnormw < 1e-6:
            #print('convergence!')
            break
    #print(rnormw)

    return y



#L2-svm
class APG_L2_m18():
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

        # p = matrix(p)  
        # q = matrix(q)
        
        # g = matrix(-np.eye(data_num))
        # h = matrix(np.zeros([data_num, 1], np.float64))

        # solvers.options['show_progress'] = False
        # sol = solvers.qp(p, q, g, h)
        # alpha_svs1 = np.array(sol['x'])

        # print(np.linalg.norm(alpha_svs1 - alpha_svs))
        # # alpha_svs = alpha_svs1

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