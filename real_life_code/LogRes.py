import numpy as np
from algorithms.clf import Clf

#error rate: 0.083
#results MR1-6: 0.0, 0.0, 0.01, 0.02, 0.0, 0.0
#results MRO1-O2: 1.0(err), 0.0

#def sigmoid(inX):
#	return 1.0 / (1 + np.exp(-inX))

def sigmoid(x):
    # avoid overflow
    return .5 * (1 + np.tanh(.5 * x))

class LogRes():
	def fit(self,dataMatIn, classLabels):
		dataMatrix = np.mat(dataMatIn)										#转换成numpy的mat
		labelMat = np.mat(classLabels).transpose()							#转换成numpy的mat,并进行转置
		m, n = np.shape(dataMatrix)											#返回dataMatrix的大小。m为行数,n为列数。
		dataMatrix = np.column_stack((dataMatrix, np.ones((m, 1))))
		n=n+1
		alpha = 0.001														#移动步长,也就是学习速率,控制更新的幅度。
		maxCycles = 30000														#最大迭代次数
		weights = np.ones((n,1))
		for k in range(maxCycles):
			h = sigmoid(dataMatrix * weights)								#梯度上升矢量化公式
			error = labelMat - h
			weights = weights + alpha * dataMatrix.transpose() * error
			if np.linalg.norm(dataMatrix.transpose() * error)<1e-6:
				print("convergence")
				break;
			if k == maxCycles-1:
				print(np.linalg.norm(dataMatrix.transpose() * error))
		#print(weights.getA())
		w = np.array(weights.getA()).flatten()
		b = w[-1]
		w = w[0:w.shape[0]-1]
		clf = Clf(w, b)
		return clf												#将矩阵转换为数组，返回权重数组