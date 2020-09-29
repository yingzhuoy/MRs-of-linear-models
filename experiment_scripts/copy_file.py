import os
import shutil

source = r'..\algorithms\Logistic_regression\GD\GD_gv.py'
for i in range(100):
	target = r'..\algorithms\Logistic_regression\GD\GD_m'+ str(i)+r'.py'
	shutil.copy(source, target)