import os
import numpy as np
from algorithms.clf import Clf

class libsvm3_24():
	def fit(self, X, y):
		y[y == 0] = -1
		m, n = X.shape
		with open('./train_set', 'w') as f:
			for i in range(m):
				f.write(str(y[i]))
				f.write(" ")
				for j in range(n):
					f.write(str(j))
					f.write(":")
					f.write(str(X[i,j]))
					if j == n - 1:
						f.write("\n")
					else:
						f.write(" ")
		
		r_v = os.system("../../libsvm/libsvm-3.24/svm-train -s 0 -t 0 -q ../../MRs-of-linear-models/run/train_set ./model")
		#print(os.getcwd())
		#r_v = os.system("../../../libsvm/libsvm-3.0/svm-train")

		#os.system("../../libsvm/libsvm-3.0/svm-predict ../../MRs-of-linear-models/run/train_set ../../MRs-of-linear-models/run/model ./output")
		
		f = open("./model", 'r')
		lines = f.readlines()
		w = 0
		b = 0
		flag = False
		for line in lines:
			if line.find("rho") != -1:
				rho = line.split(" ")
				b = float(rho[1])
			if line.find("label") != -1:
				nr_sv = line.split(" ")
				neg = float(nr_sv[1])
				if neg==-1:
					flag = True
			if line.find(":") != -1:
				res = line.split(" ")
				alpha = float(res[0])
				x_i = []
				for i in range(1, len(res)-1):
					x_i.append(float(res[i].split(":")[1]))
				x_i = np.array(x_i)
				w = w + alpha*x_i

		w = np.array(w).flatten()
		if flag == True:
			clf = Clf(-w, b)
		else:
			clf = Clf(w, -b)
		return clf

