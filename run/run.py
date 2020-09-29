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

def feval(funcName, *args):
	'''
	This function is similar to "feval" in Matlab.
	Example: feval('cos', pi) = -1.
	'''
	return eval(funcName)(*args)

if __name__ == '__main__':

	#n_train, n_test, n_redundant, n_classes, neg_class(算法是svm时 neg_class = -1, 算法是logreg时， neg_class = 0)

	datasets = CreateDataset(400,100,0,2,-1)

	for i in range(1, 123):
		f_str = 'APG_L1_m%s' %i
		print(f_str)
		lr = feval(f_str)
		test = LinearMRs(lr.fit, datasets.classification2, hyp_classification,30)
		test.MR1()
		test.MR2()
		test.MR3()
		test.MR4()
		test.MR5()
		test.MR6()
		test.MR7()
		test.MR8()
		test.MR9()


	#datasets = CreateDataset(20,5,2,0,2,-1)
	#X_train, y_train, X_test, y_test = datasets.classification()
	#lr = SVM_CVX_L1_gv()
	#clf = lr.fit(X_train, y_train)
	#err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
	#print(err)