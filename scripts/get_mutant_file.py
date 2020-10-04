import os
import shutil
import xlrd
import xlwt

data = xlrd.open_workbook(r'..\results\Newton.xls')
table = data.sheet_by_name('basic results')
#13列
to_modify = []
#12列
modified = []

mutant_num = table.nrows
for i in range(2, mutant_num):
	to_modify.append(table.cell(i, 13).value)
	modified.append(table.cell(i, 12).value)

#path = r'..\algorithms\Svm\IPM\L2'
path = r'..\algorithms\Logistic_regression\Newton'

source = path + r'\Newton_gv.py'
for i in range(mutant_num - 2):
	target = path + r'\Newton_m'+ str(i+1)+r'.py'
	shutil.copy(source, target)

rootdir = path
list = os.listdir(rootdir)
for i in range(0, len(list)-2):
	if list[i]=='__init__.py' or list[i]=='__pycache__':
		continue
	replace_word = "class " + list[i][0:-3] +"():"
	p = open(rootdir + '\\' + list[i], 'r+', encoding='utf-8')
	lines = p.readlines()
	d = ""
	for line in lines:
		c = line.replace("class Newton_gv():", replace_word)
		d+=c
	p.seek(0)
	p.truncate()
	p.write(d)
	p.close()


for i in range(mutant_num-2):
	before = to_modify[i]
	after = modified[i]
	file_path = path + r'\Newton_m' + str(i+1) + '.py'
	p = open(file_path, 'r+', encoding='utf-8')
	lines = p.readlines()
	d = ""
	for line in lines:
		c = line.replace(before, after)
		if(line.strip() == before):
			d+="#----bug----\n"
			d+="#"+before+'\n'
		d+=c
	p.seek(0)
	p.truncate()
	p.write(d)
	p.close()

rootdir = path + r'\__init__.py'

p = open(rootdir, 'w', encoding='utf-8')
p.write('from algorithms.Logistic_regression.Newton.Newton_gv import Newton_gv\n')
for i in range(148):
	p.write('from algorithms.Logistic_regression.Newton.Newton_m' + str(i+1) +' import Newton_m' + str(i+1)+'\n')
p.close()