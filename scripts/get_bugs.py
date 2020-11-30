#14 after 15 before
import xlwt
import xlrd
import xlutils.copy
import os

file_name = '../results/APG_L1.xls'
read_dir = '../algorithms/Svm/APG/L1'

rb1 = xlrd.open_workbook(file_name)
wb1 = xlutils.copy.copy(rb1)
ws1 = wb1.get_sheet('basic results')

files = os.listdir(read_dir)
for file in files:
	if file != "__pycache__" and file !="__init__.py" and file !="APG_L1_gv.py":
		no = file[8:-3]
		no = int(no)
		print(no)
		p = open(read_dir + '/' + file, encoding='utf-8')
		lines = p.readlines()
		d = ""
		flag = 0
		previousline = ''
		for line in lines:
			if "bug" in line:
				previousline.strip()
				ws1.write(no+1, 15, previousline[1:-1])
				flag = 1
				continue
			elif flag == 1:
				ws1.write(no+1, 14, line)
				break
			previousline = line

		'''
		for line in lines:
			if "bug" in line:
				flag = 1
				continue
			elif flag == 1:
				ws1.write(no, 15, line[1:-1])
				flag = 2
				continue
			elif flag == 2:
				ws1.write(no, 14, line)
				break
		'''

wb1.save(file_name)
