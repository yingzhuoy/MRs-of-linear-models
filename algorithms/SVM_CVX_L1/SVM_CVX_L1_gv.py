import numpy as np
from numpy import linalg
from algorithms.clf import Clf
# L1-svm

def projected_apg(p, q, bounds, step_size=0.1, max_iter=1000):
    m = p.shape[0]
    low, up = bounds

    theta_prev = 0
    theta_curr = 1
    gamma = 1
    w = np.zeros((m, 1))
    b = 0
    w_prev = w
    b_prev = b

    for k in range(max_iter):  # heavy on matrix operations
        
        # saving previous w
        w_old = w

        # compute loss and its gradient
        gradient = p*w + q

        # update w
        w_curr = w - step_size * gradient

        w = (1 - gamma) * w_curr + gamma * w_prev
        w_prev = w_curr

        theta_tmp = theta_curr
        theta_curr = (1 + np.sqrt(1 + 4 * theta_prev * theta_prev)) / 2
        theta_prev = theta_tmp

        gamma = (1 - theta_prev) / theta_curr   

        
        # projection
        w[w < low] = low 
        w[w > up] = up

        # stop criteria    
        rnormw = np.linalg.norm(w - w_old)/(1+linalg.norm(w))
        if  k > 1 and rnormw < 1e-6:
            print(np.linalg.norm(w-w_old))
            break

    return w



class SVM_CVX_L1_gv():
    def fit(self, X, y):
        m, n = X.shape
        X = np.column_stack((X, np.ones((m, 1))))
        y = y.astype(np.float64)
        data_num = len(y)
        C = 1.0
        kernel = np.dot(X, np.transpose(X))
        # np.outer()表示的是两个向量相乘,拿第一个向量的元素分别与第二个向量所有元素相乘得到结果的一行。
        p = np.matrix(kernel * np.outer(y, y))
        q = np.matrix(-np.ones([data_num, 1], np.float64))

        bounds = (0, C)
        
        alpha_svs = projected_apg(p, q, bounds)

        alpha_svs[alpha_svs <= 1e-4] = 0
        alpha_svs.astype(np.float64)

        y1 = np.reshape(y, (-1, 1))
        alpha1 = alpha_svs
        lambda1 = np.multiply(y1,alpha1)
        w = X.T * lambda1
        w = np.array(w).reshape(-1)
        # b = np.mean(y1-np.reshape(np.dot(w, np.transpose(X)), [-1, 1]))
        b = w[n]
        w = w[0:n]

        clf = Clf(w, b)
        return clf
