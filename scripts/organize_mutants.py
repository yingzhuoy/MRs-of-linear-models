import os.path
import sys
import xlrd
import xlwt

def setStyle(horz):
	style = xlwt.XFStyle()
	font = xlwt.Font()
	font.name = 'Avrial'
	font.height = 20*12
	style.font = font
	alignment = xlwt.Alignment()
	alignment.horz = horz
	alignment.vert = 0x02
	style.alignment = alignment
	return style


file_path = os.getcwd()+ '\\'+r'ADMM_L1.txt'
p = open(file_path, 'r', encoding='utf-8')

to_modify = []
modified = []

lines = p.readlines()
for line in lines:
	keyword = line[0:2]
	if keyword == '- ':
		line = line[1:-1]
		to_modify.append(line.strip())
	if keyword == '+ ':
		line = line[1:-1]
		modified.append(line.strip())
p.close()

mutant_num = len(to_modify)

workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('basic results')


for column in range(1,10):
	worksheet.write(1, column, 'MR%s' %column, setStyle(0x02))
for row in range(2,2+mutant_num):
	worksheet.write(row, 0, 'BUG%s' %(row-1), setStyle(0x02))

worksheet.write(1, 10, 'Detected Num', setStyle(0x02))
worksheet.write(1, 11, 'Error rate', setStyle(0x02))
worksheet.write(1, 12, 'BUG description(after)', setStyle(0x02))
worksheet.write(1, 13, 'BUG description(before)', setStyle(0x02))
worksheet.write(1, 14, 'Feature list', setStyle(0x02))
worksheet.write(1, 15, 'Error list', setStyle(0x02))
worksheet.col(10).width = 6000
worksheet.col(11).width = 6000
worksheet.col(12).width = 12000
worksheet.col(13).width = 12000
worksheet.col(14).width = 6000
worksheet.col(15).width = 6000

for row in range(2, 2 + mutant_num):
	worksheet.write(row , 12, modified[row-2], setStyle(0x04))
for row in range(2, 2 + mutant_num):
	worksheet.write(row , 13, to_modify[row-2], setStyle(0x04))

workbook.save(r'..\results\ADMM_L1.xls')

