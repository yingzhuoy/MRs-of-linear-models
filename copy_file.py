import os
import shutil

source = r'algorithms\Svm\IPM\L2\IPM_L2_gv.py'
for i in range(100):
	target = r'algorithms\Svm\IPM\L2\IPM_L2_m'+ str(i)+r'.py'
	shutil.copy(source, target)