import numpy as np
from numpy import linalg
from algorithms.clf import Clf
# L1-svm
import cvxopt
from cvxopt import matrix,solvers

def linesearch(y0, dy, p, q, tol=1e-5):
    maxit = int(np.log(1/(tol))/np.log(2))
    c1 = 1e-4; c2 = 0.9

    g0 = (p*y0 + q).T * dy
    Ly0 = y0.T*(p*y0) + q.T*(y0)

    if (g0 >= 0):
        alpha = 0; ite = 0; LY = LY0; 
        search_ok = -1;
        print('\n Need a descent direction, %2.1e  ', g0)
      
    alpha = 1e3; alphaconst = 0.5
    for ite in range(1,maxit):
        print(ite)
        if (ite == 1):
            LB = 0; UB = 1; 
        else:
            alpha = alphaconst*(LB+UB)
        y = y0 + alpha * dy
        Ly = y.T*(p*y) + q.T*(y)
        gradLy = (p*y + q)
        galp = gradLy.T * dy
        if (ite==1):
            gLB = g0; gUB = galp; 
            if (np.sign(gLB)*np.sign(gUB) > 0):
                search_ok = 3; 
                break             
        if Ly-Ly0-c1*alpha*g0 < (1e-8/max(1,abs(Ly))):
            break
      
        if (np.sign(galp)*np.sign(gUB) < 0):
            LB = alpha
            gLB = galp
        elif (np.sign(galp)*np.sign(gLB) < 0):
            UB = alpha
            gUB = galp 
    return y

def projected_apg(p, q, bounds, step_size=0.1, max_iter=10000):
    m = p.shape[0]
    low, up = bounds    

    x = np.ones((m, 1))

    v, _ = np.linalg.eigh(0.5*(p+p.T))
    L = 1/v[-1]
    step_size = 1/L - 1e-10;

    for k in range(max_iter):  # heavy on matrix operations
        
        # saving previous x
        y = x

        # compute loss and its gradient
        gradient = p*x + q

        # update w
        # t = linesearch(y, grad)
        # x = x - step_size * gradient
        x = linesearch(y, -gradient, p,q)

        # projection
        x[x < low] = low 
        x[x > up] = up

        y = x + (k-1)/(k+2) * (x - y)

        # stop criteria            
        rnormw = np.linalg.norm(y-x)/(1+linalg.norm(x))        
        if  k > 1 and rnormw < 1e-5:
            # print('convergence!')
            break

    return y



class SVM_CVX_L1_gv():
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
        p = np.matrix(np.multiply(kernel, np.outer(y, y)))              
        q = np.matrix(-np.ones([data_num, 1], np.float64))

        bounds = (0, C)
        
        alpha_svs = projected_apg(p, q, bounds)   

        # alpha_svs[alpha_svs <= 1e-4] = 0
        # alpha_svs.astype(np.float64)

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
