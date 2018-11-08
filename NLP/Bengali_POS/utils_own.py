import numpy as np

def get_poslist(fname):
	poslist=set()
	with open(fname,'r') as f:
		for l in f:
			if l.strip() and len(l.strip().split()) > 1 and l.strip().split()[0] != '#':
				#print l
				p = l.strip().split()[2]
				p = p.split(':?')[0]
				p = p.split('?')[0]
				if p == "'":
					p = l.strip().split()[3]
				poslist.add(p)
	return list(poslist)

# with open('Data/data1','r') as f:
# 	for l in f:
# 		try:
# 			if '***' in l.strip().split()[2]:
# 				print l.strip().split()
# 		except:
# 			pass

pos_list = get_poslist('Data/data1')
print pos_list, '\n'

pos_list = get_poslist('Data/data2')
print pos_list