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
from algorithms.Logistic_regression.LBFGS import *
from algorithms.Svm.APG.L1 import *
from algorithms.Svm.APG.L2 import *
from algorithms.Svm.SQP.L1 import *
from algorithms.Svm.SQP.L2 import *
from algorithms.Svm.ADMM.L1 import *
from algorithms.Svm.ADMM.L2 import *

from real_life_code.LogRes import LogRes
from real_life_code.StocGradAscent import StocGradAscent
from real_life_code.SMOSimple import SMOSimple


import numpy as np
import random

random.seed(1)
np.random.seed(1)

def save_result_to_file(file_name, row, column, err):
	rb = xlrd.open_workbook(file_name)
	wb = xlutils.copy.copy(rb)
	ws = wb.get_sheet('Filter1')
	
	ws.write(row, column + 9, err)

	wb.save(file_name)
'''
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
'''
if __name__ == '__main__':

	#n_train, n_test, n_redundant, n_classes, neg_class(算法是svm时 neg_class = -1, 算法是logreg时， neg_class = 0)
	
	import time

	datasets = CreateDataset(240,60,0,2,0)
	
	m_time = 0
	g_time = 0
	for i in range(1):
		X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
		print(X_train.shape)
		lr = LogRes()
		clf = lr.fit(X_train, y_train)
		err, pred, conf = sig_classification(clf.coef_, clf.intercept_, X_test, y_test)
		print(err)
		
		test = LinearMRs(lr.fit, datasets.create_dataset, sig_classification, 1)
		# print(test.MR1())
		#print(test.MR2())
		#print(test.MR3())
		#print(test.MR4()) 
		#print(test.MR5())
		#print(test.MR6())
		print(test.MR7())
		#print(test.MR8())
		
		#m_time = m_time + t2 - t1
		#print('m73')
		#print(t2 - t1)

		#t1 = time.time()
		#lr = GD_gv()
		#lr.fit(X_train, y_train)
		#t2 = time.time()
		#g_time = g_time + t2 - t1
		#print('gv')
		#print(t2-t1)
	#print(m_time)
	#print(g_time)



	'''
	for j in range(3,11):
		np.random.seed(j)
		xls_path = '../results/SQP_L2.xls'
		datasets = CreateDataset(240,60,0,2,-1)
		X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
		for i in range(1, 117):
			f_str = 'lr = SQP_L2_m%s()' %i
			print(f_str)
			exec(f_str)
			clf = lr.fit(X_train, y_train)
			err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
			save_result_to_file(xls_path, i, j, err)

		lr = SQP_L1_gv()
		clf = lr.fit(X_train, y_train)
		err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
		save_result_to_file(xls_path, 89, j, err)
	'''

	'''
	datasets = CreateDataset(240,60,0,2,-1)
	print("SQP_L2_gv()")
	lr = SQP_L2_m91()
	X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
	clf = lr.fit(X_train, y_train)
	err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
	test = LinearMRs(lr.fit, datasets.create_dataset, hyp_classification,10)
	print(err)
	#print(test.MR1())
	#print(test.MR2())
	#print(test.MR3())
	#print(test.MR4()) 
	#print(test.MR5())
	#print(test.MR6())
	print(test.MR7())
	#print(test.MR8())
	'''
	'''
	xls_path = '../results/SQP_L2.xls'

	for j in range(96, 101):

		res_list = []
		feature_list = []
		single_res_list = []
		err = 0
		
		datasets = CreateDataset(240,60,0,2,-1)
		#不同算法相应的调用部分要改一下
		f_str = 'lr = SQP_L2_m%s()' %j
		#f_str = 'lr = SQP_L1_gv()'
		print(f_str)
		exec(f_str)

		#calculate error rate of this mutant
		try:
			X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
			clf = lr.fit(X_train, y_train)
			err, pred, conf = hyp_classification(clf.coef_, clf.intercept_, X_test, y_test)
		except ValueError as e1:
			continue
		except IndexError as e3:
			continue
		except TypeError as e4:
			continue
		else:
			pass
		finally:
			pass
		print(err)

		#test MRs on the mutant
		test = LinearMRs(lr.fit, datasets.create_dataset, hyp_classification,100)
		for i in range(1, 9):
			try:
				exec('r%s, f%s, s%s = test.MR%s()' %(i, i, i, i))
				exec('f_str2 = res_list.append(r%s)' %i)
				exec('feature_list.append(f%s)' %i)
				exec('single_res_list.append(s%s)' %i)
				exec('print(r%s)' %i)
			except BaseException as e1:
				f_str2 = res_list.append('base_exception_err')
			except FloatingPointError as e2:
				f_str2 = res_list.append('floating_point_err')
			except IndexError as e3:
				f_str2 = res_list.append('index_err')
			except ValueError as e4:
				f_str2 = res_list.append('value_err')
			except ValueError as e5:
				f_str2 = res_list.append('type_err')
			else:
				pass
			finally:
				pass
				
		save_result_to_file(xls_path, j+1, res_list, feature_list, single_res_list, err)
		'''