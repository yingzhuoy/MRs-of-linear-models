import sys
import xlwt
import xlrd
import xlutils.copy
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
np.random.seed(1)

def save_result_to_file(file_name, row, res_list, feature_list, single_res_list, err):
	rb = xlrd.open_workbook(file_name)
	wb = xlutils.copy.copy(rb)
	ws = wb.get_sheet('basic results')
	
	str_feature = ''
	str_single_res = ''
	feature_list = str(feature_list)
	single_res_list = str(single_res_list)
	
	str_feature = str_feature.join(feature_list)
	str_single_res = str_single_res.join(single_res_list)

	for column in range(7):
		ws.write(row, column+1, res_list[column])
	ws.write(row, 11, err)
	ws.write(row, 14, str_feature)
	ws.write(row, 15, str_single_res)

	wb.save(file_name)

if __name__ == '__main__':

	#n_train, n_test, n_redundant, n_classes, neg_class(算法是svm时 neg_class = -1, 算法是logreg时， neg_class = 0)
	
	'''
	datasets = CreateDataset(240,60,0,2,0)
	lr = Newton_m1()
	X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
	clf = lr.fit(X_train, y_train)
	err, pred, conf = sig_classification(clf.coef_, clf.intercept_, X_test, y_test)
	test = LinearMRs(lr.fit, datasets.create_dataset, sig_classification,100)
	print(err)
	for i in range(1, 10):
		exec('r%s, f%s, s%s = test.MR%s()' %(i, i, i, i))
		exec('print(r%s)' %i)
	'''

	for j in range(1, 146):
		res_list = []
		feature_list = []
		single_res_list = []

		datasets = CreateDataset(240,60,0,2,0)
		f_str = 'lr = Newton_m%s()' %j
		print(f_str)
		exec(f_str)

		#calculate error rate of this mutant
		X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
		clf = lr.fit(X_train, y_train)
		err, pred, conf = sig_classification(clf.coef_, clf.intercept_, X_test, y_test)

		#test MRs on the mutant
		test = LinearMRs(lr.fit, datasets.create_dataset, sig_classification,100)
		for i in range(1, 8):
			exec('r%s, f%s, s%s = test.MR%s()' %(i, i, i, i))
			exec('f_str2 = res_list.append(r%s)' %i)
			exec('feature_list.append(f%s)' %i)
			exec('single_res_list.append(s%s)' %i)
			exec('print(r%s)' %i)
		save_result_to_file(r'..\results\Newton.xls', j+1, res_list, feature_list, single_res_list, err)






	'''
	for i in range(1, 123):
		f_str = 'APG_L1_m%s' %i
		print(f_str)
		lr = feval(f_str)
		test = LinearMRs(lr.fit, datasets.create_dataset, hyp_classification,30)
		test.MR1()
		test.MR2()
		test.MR3()
		test.MR4()
		test.MR5()
		test.MR6()
		test.MR7()
		test.MR8()
		test.MR9()
	'''

	#datasets = CreateDataset(20,5,2,0,2,-1)
	#X_train, y_train, X_test, y_test = datasets.classification()
	#lr = SVM_CVX_L1_gv()
	#clf = lr.fit(X_train, y_train)
	#err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
	#print(err)