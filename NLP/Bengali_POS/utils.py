import os, sys, re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
#featlist= ['Pos', 'Case', 'Polarity', 'Animacy', 'Mood', 'Definiteness', 'Verbform', 'Number', 'Person', 'Tense', 'Aspect', 'Honorific', 'Proper', 'Emphatic', 'Prontype']

def get_poslist(fname):
	poslist=set()
	with open(fname,'r') as f:
		for l in f:
			if l.strip() and len(l.strip().split()) > 1 and l.strip().split()[0] != '#':
				p = clean_pos(l)
				poslist.add(p)
	return list(poslist)

def clean_pos(l):
 	p= l.strip().split()[2]
	p = p.split(':?')[0]
	p = p.split('?')[0]
	if p == "'":
		p = l.strip().split()[3]
	return p


def load_words(data_type = 'data1'):
	X = []
	X1=[]

	Y = []
	Y1= []

	word_embeddings=[]
	word_embeddings2={}

	lc=0
	words=[]


	#word_idx=['ADV', 'NOUN', 'ADP', 'PRON', 'SCONJ', 'PROPN', 'DET', 'SYM', 'INTJ', 'PART', 'PUNCT', 'VERB', 'X', 'AUX', 'CCONJ', 'NUM', 'ADJ']
	#pos_idx=get_poslist('bn.data')
	poslist=get_poslist('Data/'+data_type)

	
	#with open('/home/ayan/morphanalysis/pos_morph_data/PHASE_2/embeddings.txt') as f1:
	with open('Data/embeddings.txt') as f1:
		for line1 in tqdm(f1):
			word_embeddings2[line1.strip().split(' ')[0]]=[float(i) for i in line1.strip().split()[1:]]
	for key in word_embeddings2:
		word_embeddings.append(word_embeddings2[key])

	#with open('/home/ayan/morphanalysis/pos_morph_data/PHASE_2/bn.data.phase_1_2') as f:
	with open('Data/'+data_type) as f:
			for line in tqdm(f):
				if line.strip() != '' and len(line.strip().split()) > 1  and line.strip().split()[0] != '#':
					#if line.strip() == ' SYM':
					#	continue
					word=line.strip().split()[1]
					'''
					if word not in word_embeddings2:
						print 'oov found'
						# generate word embedding
						with open('query.txt','w') as fout:
							fout.write(word)
						os.system('fastText-master/build/fasttext print-word-vectors cc.bn.300.bin < query.txt > outvec.txt')
						with open('outvec.txt','r') as f:
							for l in f:
								word_embeddings2[l.strip().split(' ')[0]]=[float(i) for i in l.strip().split()[1:]]
								word_embeddings.append(word_embeddings2[l.strip().split(' ')[0]])
					'''
					
					if word in word_embeddings2:
						words.append(word)
		 				X1.append(word_embeddings2.keys().index(word))
						Y1.append(poslist.index(clean_pos(line)))
						#print word, word_embeddings2.keys().index(word), poslist.index(line.strip().split()[2])
					else:
						print 'word = ', word, line
						#line.strip().split('\t')[4])
	 			else:
					if len(X1)==0:
						continue
					lc+=1
					
					X.append(np.asarray(X1))
					Y.append(np.asarray(Y1))
						
					words=[]
					X1=[]
					Y1=[]
	
 	word_embeddings=np.asarray(word_embeddings)
	unk=np.mean(word_embeddings, axis=0)
	blank=np.zeros(300)
	#print word_embeddings.shape, blank.shape, X.shape, Y.shape, unk.shape
	word_embeddings=np.vstack([word_embeddings,unk,blank])
	#print len(word_embeddings)
 	return word_embeddings, poslist,X,Y

if __name__ == "__main__":
	word_embeddings, poslist,X,Y=load_words('data1')
	print X, Y
