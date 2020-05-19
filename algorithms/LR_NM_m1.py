import sys
sys.path.append(r'..')

#gradAscent
from numpy import *
import numpy as np
from algorithms.clf import Clf

class LR_NM_m1():
    """docstring for LogReg_NewtonMethod_GoldenVersion"""
    def p1(self, x):
        return exp(x) / (1 + exp(x))

    def delta(self, beta, dataMat, labelMat):
        fderiv = - np.sum(np.multiply(dataMat , (labelMat - self.p1(dataMat * beta.T))), axis = 0)
        sderiv = np.sum(np.multiply(np.multiply(dataMat , dataMat) ,np.multiply(self.p1(dataMat * beta.T), (1 - self.p1(dataMat * beta.T)))))
        return fderiv / sderiv

    #newtonMethod
    def fit(self, dataMatIn, classLabels):
        dataMat = mat(dataMatIn)             #convert to NumPy matrix
        labelMat = mat(classLabels).transpose() #convert to NumPy matrix
        m, n = shape(dataMat)

        dataMat = np.column_stack((dataMat, np.ones((m,1))))
        w = mat(ones(n+1))
        d = self.delta(w, dataMat, labelMat)
        for _ in range(1000):
            norm_d = linalg.norm(d)
            '''print('The {!s}th iteration, the norm of is newton direction is {!s}'.format(
                _, norm_d))'''
            if norm_d < 1e-3:
                break
            else:
                #-----bug1-----
                w = w - d + 0.001
                #w = w - d
                d = self.delta(w, dataMat, labelMat)
        w = np.array(w).flatten()
        b = w[n]
        w = w[0:n]
        clf = Clf(w, b)
        return clf