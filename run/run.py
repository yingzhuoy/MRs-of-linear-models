import sys
sys.path.append(r'..')


from datasets.create_datasets import CreateDataset
from test_programs.test_program import sig_classification, hyp_classification
from MRs.linear_MRs import LinearMRs

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC

from algorithms.LR_GA import *
from algorithms.LR_NM import *
#from algorithms.svm_test import SVM

#from algorithms.SVM_SMO_gv import SVM_SMO_gv
#from algorithms.SVM_CVX_L1_gv import SVM_CVX_L1_gv
#from algorithms.SVM_CVX_gv1 import SVM_CVX_gv1
#from algorithms.SVM_CVX_m1 import SVM_CVX_m1
#from algorithms.SVM_CVX_m2 import SVM_CVX_m2
#from algorithms.SVM_CVX_m3 import SVM_CVX_m3
#from algorithms.SVM_CVX_m4 import SVM_CVX_m4
#from algorithms.SVM_CVX_m5 import SVM_CVX_m5
#from algorithms.clf import Clf

import numpy as np
np.random.seed(1)

if __name__ == '__main__':

	#n_train, n_test, n_features, n_redundant, n_classes, neg_class(算法是svm时 neg_class = -1, 算法是logreg时， neg_class = 0)
	
	datasets = CreateDataset(400,100,10,0,2,0)
	
	lr = LR_NM_m46()
	test = LinearMRs(lr.fit, datasets.classification, sig_classification, 50)

	test.MR1()
	test.MR2()
	test.MR3()
	test.MR4()
	test.MR5()
	test.MR6()
	test.MR7()
	test.MR8()
	test.MR9()
	#lr = LR_GA_gv()
	#test = LinearMRs(lr.fit, datasets.classification, sig_classification, 100)
	#test.MR7()



	#如果要使用sklearn库，则直接
	#print("LR_GA_gv")
	# lr = LR_NM_m3()
	# #lr = LogisticRegression(solver='newton-cg', penalty='l2')
	# #lr = LogisticRegression()
	# test = LinearMRs(lr.fit, datasets.classification, sig_classification, 500)
	# #test.MR1()
	# #test.MR2()
	# #test.MR3()
	# #test.MR4()
	# #test.MR5()
	# #test.MR6()
	# #test.MR7()
	# test.MR8()
	# test.MR9()



	#datasets = CreateDataset(400,100,10,0,2,-1)
	#X_train, y_train, X_test, y_test = datasets.classification()
	# svm = SVM_CVX_gv()
	# clf = svm.fit(X_train, y_train)
	#print(svm.w)
	#print(svm.b)
	#err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
	#print(clf.coef_)
	#print(err)
	#print(pred)
	#print(pred)
	#print(conf)

	#svm = LinearSVC(loss = 'hinge', max_iter=100000)
	#fit(待测试算法的fit), create_dataset(产生数据集的函数), test_program(测试数据集是用sigmoid还是超平面划分), itr_cnt(测试的循环次数)
	#err, pred, conf