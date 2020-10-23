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
	datasets = CreateDataset(240,60,0,2,0)
	lr = lbfgs_gv()
	X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
	clf = lr.fit(X_train, y_train)
	err, pred, conf = sig_classification(clf.coef_, clf.intercept_, X_test, y_test)
	test = LinearMRs(lr.fit, datasets.create_dataset, sig_classification,30)
	print(err)
	print(test.MR1())
	print(test.MR2())
	print(test.MR3())
	print(test.MR4())
	print(test.MR5())
	print(test.MR6())
	print(test.MR8())
	print(test.MR7())
	'''
	
	
	#不同算法对应不同的表格路径，mutant的数量也不相同，自己设定
	xls_path = '../results/LBFGS.xls'
	mutant_num = 203

	#21, 26\27, 35, 163-165
	#mutant的数量自己设定，会写在对应的行，每跑完一个mutant都会在表格相应位置记录下来，所以就算中途程序停止也没问题
	#比如要跑第10到第20个mutant，则设定for j in range(10,21)
	for j in range(76, 77):

		res_list = []
		feature_list = []
		single_res_list = []
		err = 0
		
		datasets = CreateDataset(240,60,0,2,0)
		#不同算法相应的调用部分要改一下
		f_str = 'lr = lbfgs_m%s()' %j
		print(f_str)
		exec(f_str)

		#calculate error rate of this mutant
		try:
			X_train, y_train, X_test, y_test, feature_num = datasets.create_dataset()
			clf = lr.fit(X_train, y_train)
			err, pred, conf = sig_classification(clf.coef_, clf.intercept_, X_test, y_test)
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

		#test MRs on the mutant
		test = LinearMRs(lr.fit, datasets.create_dataset, sig_classification,100)
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