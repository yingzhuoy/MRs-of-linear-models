import sys
sys.path.append(r'..')

#gradAscent
from numpy import *
import numpy as np
from algorithms.clf import Clf

class LR_GA_m1():
        
    def sigmoid(self, inX):
        return 1/(1+exp(-inX))

    #gradAscent
    def fit(self, dataMatIn, classLabels):
        dataMat = mat(dataMatIn)             #convert to NumPy matrix
        labelMat = mat(classLabels).transpose() #convert to NumPy matrix
        m, n = shape(dataMat)

        dataMat = np.column_stack((dataMat, np.ones((m,1))))

        #print(dataMat)
        alpha = 0.001
        maxCycles = 500
        w = ones(( n + 1 ,1))
        for k in range(maxCycles):              #heavy on matrix operations
            h = self.sigmoid(dataMat*w)     #matrix mult
            error = (labelMat - h)              #vector subtraction
            #------bug1--------
            w = w + alpha * dataMat.transpose()* error + 0.01 #matrix mult
            #w = w + alpha * dataMat.transpose()* error #matrix mult

        w = np.array(w).flatten()
        b = w[n]
        w = w[0:n]
        clf = Clf(w, b)
        print(w)
        #w: n*1 array b: number
        return clf