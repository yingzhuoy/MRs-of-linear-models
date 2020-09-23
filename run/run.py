import sys

sys.path.append(r'..')


from datasets.create_datasets import CreateDataset
from test_programs.test_program import sig_classification, hyp_classification
from MRs.linear_MRs import LinearMRs

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC

from algorithms.Logistic_regression.GD import *
from algorithms.Logistic_regression.Newton import *
from algorithms.Svm.APG.L1 import *
from algorithms.Svm.APG.L2 import *
from algorithms.Svm.IPM.L1 import *
from algorithms.Svm.IPM.L2 import *
import numpy as np
#np.random.seed(1)

if __name__ == '__main__':

	#n_train, n_test, n_features, n_redundant, n_classes, neg_class(算法是svm时 neg_class = -1, 算法是logreg时， neg_class = 0)
	
	datasets = CreateDataset(80,20,2,0,2,-1)
	#datasets = CreateDataset(400,100,10,0,2,0)
	'''
	print("APG L1")
	lr = APG_L1_gv()
	test = LinearMRs(lr.fit, datasets.classification2, hyp_classification,100)
	test.MR7()
	test.MR8()
	test.MR9()
	'''
	#print("APG L2")
	#lr = APG_L2_gv()
	#test = LinearMRs(lr.fit, datasets.classification2, hyp_classification,100)
	#test.MR7()
	#test.MR8()
	#test.MR9()
	'''
	print("IPM L1")
	lr = IPM_L1_gv()
	test = LinearMRs(lr.fit, datasets.classification2, hyp_classification,100)
	test.MR7()
	test.MR8()
	test.MR9()

	print("IPM L2")
	lr = IPM_L2_gv()
	test = LinearMRs(lr.fit, datasets.classification2, hyp_classification,100)
	test.MR7()
	test.MR8()
	test.MR9()
	'''
	print("GD")
	lr = GD_gv()
	test = LinearMRs(lr.fit, datasets.classification, hyp_classification,100)
	test.MR7()
	#test.MR8()
	#n_samples = random.randint(50,200) test.MR9()
	'''
	print("Newton")
	lr = Newton_gv()
	test = LinearMRs(lr.fit, datasets.classification2, hyp_classification,100)
	test.MR7()
	test.MR8()
	test.MR9()
	'''

	#test = LinearMRs(lr.fit, datasets.classification, sig_classification, 50)

	#test.MR1()
	#test.MR2()
	#test.MR3()
	#test.MR4()
	#test.MR5()
	#test.MR6()
	#test.MR7()
	#test.MR8()
	#test.MR9()
	#lr = LR_GA_gv()
	#test = LinearMRs(lr.fit, datasets.classification, sig_classification, 100)
	#test.MR7()
	
	#datasets = CreateDataset(20,5,2,0,2,-1)
	#X_train, y_train, X_test, y_test = datasets.classification()
	#lr = SVM_CVX_L2_gv()
	#clf = lr.fit(X_train, y_train)
	#err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
	#print(err)