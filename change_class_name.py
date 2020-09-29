import os.path

rootdir = r'algorithms\Svm\IPM\L2'
list = os.listdir(rootdir)

for i in range(0, len(list)):
	if list[i]=='__init__.py' or list[i]=='__pycache__':
		continue
	replace_word = "class " + list[i][0:-3] +"():"
	p = open(rootdir + '\\' + list[i], 'r+', encoding='utf-8')
	lines = p.readlines()
	d = ""
	for line in lines:
		c = line.replace("class   IPM_L2_gv():", replace_word)
		d+=c
	p.seek(0)
	p.truncate()
	p.write(d)
	p.close()
