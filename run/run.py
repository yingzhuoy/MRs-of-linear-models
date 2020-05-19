import sys
sys.path.append(r'..')


from datasets.create_datasets import CreateDataset
from test_programs.test_program import sig_classification, hyp_classification
from MRs.linear_MRs import LinearMRs

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from algorithms.LR_GA_gv import LR_GA_gv
from algorithms.LR_GA_m1 import LR_GA_m1
from algorithms.LR_GA_m2 import LR_GA_m2
from algorithms.LR_GA_m3 import LR_GA_m3
from algorithms.LR_GA_m4 import LR_GA_m4
from algorithms.LR_GA_m5 import LR_GA_m5

from algorithms.LR_NM_gv import LR_NM_gv
from algorithms.LR_NM_m1 import LR_NM_m1
from algorithms.LR_NM_m2 import LR_NM_m2
from algorithms.LR_NM_m3 import LR_NM_m3
from algorithms.LR_NM_m4 import LR_NM_m4

from algorithms.SVM_SMO_gv import SVM_SMO_gv

if __name__ == '__main__':

	#n_train, n_test, n_features, n_redundant, n_classes, neg_class(算法是svm时 neg_class = -1, 算法是logreg时， neg_class = 0)
	datasets = CreateDataset(400,100,10,0,2,0)


	ga_gv = LR_GA_gv()
	#fit(待测试算法的fit), create_dataset(产生数据集的函数), test_program(测试数据集是用sigmoid还是超平面划分), itr_cnt(测试的循环次数)
	test = LinearMRs(ga_gv.fit, datasets.classification, sig_classification, 1000)

	#如果要使用sklearn库，则直接
	#test = LinearMRs(LogisticRegression(solver='lbfgs').fit, datasets.classification, sig_classification, 1000)

	test.MR1()
	test.MR2()
	test.MR3()
	test.MR4()
	test.MR5()
	test.MR6()
	test.MR8()
	test.MR9()