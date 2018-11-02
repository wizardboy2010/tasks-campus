import numpy as np

def get_poslist(filename):
    with open(filename, 'r') as f:
        pos = [l.strip().split(' ')[2] for l in f if(l.strip() != '' and l.strip().split(' ')[0] != '#')]
    return set(pos)

with open('Data/data1','r', encoding = 'utf-8') as f:
	for l in f:
		print(l.strip())

#pos_list = get_poslist('Data/data1')
#print(pos_list)