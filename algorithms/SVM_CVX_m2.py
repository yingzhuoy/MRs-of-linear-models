import numpy as np
from numpy import linalg
import cvxopt
from cvxopt import matrix,solvers
from algorithms.clf import Clf

class SVM_CVX_m2():
    def fit(self, X, y):
        y = y.astype(np.float64)
        data_num = len(y)
        penaltyparC = 1
        kernal = np.dot(X, np.transpose(X)) + np.diag(np.ones(data_num, np.float64)) * (1 / penaltyparC)
        p = matrix(kernal * np.outer(y, y))  #np.outer()表示的是两个向量相乘,拿第一个向量的元素分别与第二个向量所有元素相乘得到结果的一行。
        q = matrix(-np.ones([data_num, 1], np.float64))
        g = matrix(-np.eye(data_num))
        h = matrix(np.zeros([data_num, 1], np.float64))

        a = matrix(y, (1, data_num))

        #------bug2-----
        b = matrix(10.0)
        #b = matrix(0.)
        solvers.options['show_progress'] = False
        # if n_num_ == 41:
        #     stop = 1
        sol = solvers.qp(p, q, g, h, a, b)
        alpha_svs = np.array(sol['x'])
        alpha_svs[alpha_svs <= 1e-4] = 0
        alpha_svs.astype(np.float64)

        y1 = np.reshape(y,(-1,1))
        alpha1=alpha_svs
        lambda1=y1*alpha1
        w=np.sum(lambda1*X,axis=0)
        b=np.mean(y1-np.reshape(np.dot(w, np.transpose(X)), [-1, 1]))

        clf = Clf(w, b)
        return clf