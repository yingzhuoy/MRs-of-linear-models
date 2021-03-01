import numpy as np
from algorithms.clf import Clf

#error rate: 0.6177
#results MR1-6: 0.0, 0.0, 0.76, 0.38, 0.0, 0.0
#results MRO1-O2: 0.99, 0.0

def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

class LogRes():
	def fit(self,dataMatIn, classLabels):
		dataMatrix = np.mat(dataMatIn)										#转换成numpy的mat
		labelMat = np.mat(classLabels).transpose()							#转换成numpy的mat,并进行转置
		m, n = np.shape(dataMatrix)											#返回dataMatrix的大小。m为行数,n为列数。
		alpha = 0.001														#移动步长,也就是学习速率,控制更新的幅度。
		maxCycles = 500														#最大迭代次数
		weights = np.ones((n,1))
		for k in range(maxCycles):
			h = sigmoid(dataMatrix * weights)								#梯度上升矢量化公式
			error = labelMat - h
			weights = weights + alpha * dataMatrix.transpose() * error
		#print(weights.getA())
		clf = Clf(np.array(weights.getA()).reshape(-1), 0)
		return clf												#将矩阵转换为数组，返回权重数组