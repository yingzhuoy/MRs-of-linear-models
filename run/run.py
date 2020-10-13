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
from algorithms.Svm.ADMM.L1 import *
from algorithms.Svm.ADMM.L2 import *

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
	
	#MR1-8分别存储在对应的列数，如果要记录特定MR的结果修改这里
	for column in range(8):
		ws.write(row, column+1, res_list[column])

	ws.write(row, 11, err)
	ws.write(row, 14, str_feature)
	ws.write(row, 15, str_single_res)

	wb.save(file_name)

if __name__ == '__main__':

	#n_train, n_test, n_redundant, n_classes, neg_class(算法是svm时 neg_class = -1, 算法是logreg时， neg_class = 0)
	'''
	datasets = CreateDataset(240,60,0,2,-1)
	lr = ADMM_L1_gv()
	X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
	clf = lr.fit(X_train, y_train)
	err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
	test = LinearMRs(lr.fit, datasets.create_dataset, hyp_classification,30)
	print(err)
	print(test.MR8())
	print(test.MR7())
	'''
	
	#不同算法对应不同的表格路径，mutant的数量也不相同，自己设定
	xls_path = '../results/ADMM_L1.xls'
	mutant_num = 203

	
	#mutant的数量自己设定，会写在对应的行，每跑完一个mutant都会在表格相应位置记录下来，所以就算中途程序停止也没问题
	#比如要跑第10到第20个mutant，则设定for j in range(10,21)
	for j in range(1, mutant_num + 1):
		res_list = []
		feature_list = []
		single_res_list = []
		err = 0
		
		datasets = CreateDataset(240,60,0,2,-1)
		#不同算法相应的调用部分要改一下
		f_str = 'lr = ADMM_L1_m%s()' %j
		print(f_str)
		exec(f_str)

		#calculate error rate of this mutant
		X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
		clf = lr.fit(X_train, y_train)
		err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)

		#test MRs on the mutant
		test = LinearMRs(lr.fit, datasets.create_dataset, hyp_classification,100)
		for i in range(8, 9):
			exec('r%s, f%s, s%s = test.MR%s()' %(i, i, i, i))
			exec('f_str2 = res_list.append(r%s)' %i)
			exec('feature_list.append(f%s)' %i)
			exec('single_res_list.append(s%s)' %i)
			exec('print(r%s)' %i)
		
		save_result_to_file(xls_path, j+1, res_list, feature_list, single_res_list, err)