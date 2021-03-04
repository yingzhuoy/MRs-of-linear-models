import numpy as np
import random
from algorithms.clf import Clf

#error rate:0.117
#MR1-6:0.12, 0.11, 0.08, 0.12, 0, 0
#MRO1-MRO2:1.0, 0.89


def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

class StocGradAscent(object):
	"""docstring for StocGradAscent"""
	def fit(self, dataMatrix, classLabels, numIter=150):
		m,n = np.shape(dataMatrix)												#返回dataMatrix的大小。m为行数,n为列数。
		dataMatrix = np.column_stack((dataMatrix, np.ones((m, 1))))
		n=n+1
		weights = np.ones(n)   													#参数初始化
		weights_array = np.array([])											#存储每次更新的回归系数
		for j in range(numIter):											
			dataIndex = list(range(m))
			for i in range(m):			
				alpha = 4/(1.0+j+i)+0.01   	 									#降低alpha的大小，每次减小1/(j+i)。
				randIndex = int(random.uniform(0,len(dataIndex)))				#随机选取样本
				h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))		#选择随机选取的一个样本，计算h
				error = classLabels[dataIndex[randIndex]] - h 								#计算误差
				weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]   	#更新回归系数
				weights_array = np.append(weights_array,weights,axis=0) 		#添加回归系数到数组中
				del(dataIndex[randIndex]) 										#删除已经使用的样本
		weights_array = weights_array.reshape(numIter*m,n) 						#改变维度
		#clf = Clf(np.array(weights).reshape(-1), 0)
		w = np.array(weights).flatten()
		b = w[-1]
		w = w[0:w.shape[0]-1]
		clf = Clf(w, b)
		return clf												#将矩阵转换为数组，返回权重数组					