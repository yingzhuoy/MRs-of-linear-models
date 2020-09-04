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
from algorithms.SVM_CVX_L1 import *
from algorithms.SVM_CVX_L2 import *
import numpy as np
#np.random.seed(1)

if __name__ == '__main__':

	#n_train, n_test, n_features, n_redundant, n_classes, neg_class(算法是svm时 neg_class = -1, 算法是logreg时， neg_class = 0)
	
	#datasets = CreateDataset(400,100,10,0,2,-1)
	datasets = CreateDataset(400,100,10,0,2,0)
	
	#lr = SVM_CVX_L2_m12()
	lr = LR_GA_m1()

	#test = LinearMRs(lr.fit, datasets.classification, hyp_classification, 50)
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
	
	#datasets = CreateDataset(20,5,2,0,2,-1)
	#X_train, y_train, X_test, y_test = datasets.classification()
	#lr = SVM_CVX_L2_gv()
	#clf = lr.fit(X_train, y_train)
	#err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
	#print(err)