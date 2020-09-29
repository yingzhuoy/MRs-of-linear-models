import os.path

file_path = r'./Newton_10.txt'
p = open(file_path, 'r+', encoding='utf-8')

to_modify = {}
modified = {}

lines = p.readlines()
for line in lines:
	print(line[0:1])
