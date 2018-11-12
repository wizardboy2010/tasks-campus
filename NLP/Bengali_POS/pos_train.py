import os, sys
import tensorflow as tf
import numpy as np
from utils import *


word_embeddings, poslist,X,Y = load_words('data2')

hm_epochs = 500
emb_size=300
hidden_size=300
layers_count=3
class_counts=len(poslist)

vocab_size=word_embeddings.shape[0]
print 'vocab_size =', vocab_size
lr=0.0001
learning_rate = lr
batch_size = 1
word_embedding_weights=word_embeddings
word_embedding_weights=tf.cast(tf.reshape(word_embedding_weights, [len(word_embedding_weights),len(word_embedding_weights[0])]), tf.float32)
print 'word_embedding_weights.shape =', word_embedding_weights.shape
transition_params=tf.get_variable('transition_params',[len(poslist), len(poslist)])



cells=[]
for i in range(layers_count):
	cells.append(tf.nn.rnn_cell.BasicRNNCell(hidden_size))
stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

cells1=[]
for i in range(layers_count):
	cells1.append(tf.nn.rnn_cell.BasicRNNCell(hidden_size))
stacked_cells1 = tf.nn.rnn_cell.MultiRNNCell(cells1, state_is_tuple=True)

def build_model(inpx=None):
	
	word_embeddings = tf.Variable(word_embedding_weights, trainable = False)
	embed1 = tf.nn.embedding_lookup(word_embeddings, inpx)
	lstm_inp=tf.reshape(embed1,[1,tf.shape(embed1)[1],emb_size])
	
	output, state = tf.nn.bidirectional_dynamic_rnn(stacked_cells, stacked_cells1, lstm_inp,dtype=tf.float32)
	
	concat=tf.concat([output[0], output[1]], axis=2)
	concat1=tf.concat([concat, lstm_inp], axis=2)
	
	output=tf.layers.dense(concat1, class_counts)
	return output

def iterateminibatches(X,Y,batch_size,shuffle=False):
	assert len(X) == len(Y)
	if shuffle:
		indices = np.arange(len(Y))
		#np.random.shuffle(indices)
	for start_idx in range(0, len(Y) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		XYr={}
		XYr['X']=[X[i] for i in excerpt]
		
		XYr['Pos']=[Y[i] for i in excerpt]
		yield XYr

input1={'input_x':tf.placeholder(tf.int32, shape=None)}
input1['Pos']=tf.placeholder(tf.int32, shape=None)


network=build_model(input1['input_x'])

labels=input1['Pos']
sqlen=tf.shape(labels)[1]
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(network, labels, tf.reshape(sqlen, [1]))
transition_params=transition_params
loss1 = tf.reduce_mean(-log_likelihood)
loss_final=loss1
optimizer = tf.train.AdamOptimizer(learning_rate)
update = optimizer.minimize(loss_final)
	
init=tf.global_variables_initializer()
saver = tf.train.Saver()
session = tf.Session()
session.run(init)
#saver.restore(session, "/home/ayan/morphanalysis/models.bn/pos/90/my-model.ckpt-90")
for epoch in xrange(hm_epochs):
	batch_count=0.0
	epoch_loss = 0
	if epoch > 0 and epoch % 20 == 0:
		learning_rate=lr/(epoch/10)
	epoch_accuracy = 0.0
	for batch in iterateminibatches(X,Y,1,shuffle=True):
		#if len(batch['X'][0]) == 0:
		#	continue
		epoch_xy = batch
		feeddict={}
		feeddict[input1['input_x']]=epoch_xy['X']
		#for feat in featurelistsdict:
		feeddict[input1['Pos']]=epoch_xy['Pos']
		#print batch, len(batch['X'][0])
		u, l = session.run([update, loss_final], feed_dict=feeddict)
		
		#acc = session.run(transition_params, feed_dict=feeddict)
		#print acc
		#exit(0)
		#for feat in featurelistsdict:
		#	epoch_accuracy[feat]+=acc[feat]
		#print u, l
		epoch_loss+=l
		batch_count+=1
		if batch_count % 1000 == 0:
			print batch_count
		#print batch_count, 'accuracy = ', acc
	print 'Epoch loss = ', epoch_loss, 'Epoch = ', epoch
	#for feat in featurelistsdict:
	#	epoch_accuracy[feat]=epoch_accuracy[feat]/batch_count

	#print epoch_accuracy
	#if epoch % 2 == 0:
		#modelpath=os.path.join('/home/ayan/morphanalysis/pos_morph_data/PHASE_2/models',str(epoch))
	dirname = 'models/pos_crf_concat_word_embeddings_ILMT_check'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	modelpath=os.path.join(dirname,str(epoch))
	os.system('mkdir '+modelpath)
	saver.save(session,os.path.join(modelpath,'my-model.ckpt'), global_step=epoch)
	# _, acc = session.run([accuracy], feed_dict={input_x: epoch_x, input_y: epoch_y})
	# epoch_accuracy += acc
