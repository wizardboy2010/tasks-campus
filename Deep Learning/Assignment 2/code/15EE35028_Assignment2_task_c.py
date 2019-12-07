from zipfile import ZipFile
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LogisticRegression
import pickle as pkl

'''load your data here'''

assert len(sys.argv) == 2
mode = sys.argv[1]
assert mode[:2] == '--'
mode = mode[2:]

class DataLoader(object):
    def __init__(self):
        DIR = '../data/'
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

    def create_batches(self):
        pass


def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        cumulative_loss = 0
        data, label = transform(data,label)
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(data)
        loss = softmax_cross_entropy(output, label)
        cumulative_loss += nd.sum(loss).asscalar()
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1], cumulative_loss

def evaluate_accuracy_train(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data, label = transform(data,label)
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

class MLP(gluon.Block):
    def __init__(self,  **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(1024)
            self.dense1 = gluon.nn.Dense(512)
            self.dense2 = gluon.nn.Dense(256)
            self.dense3 = gluon.nn.Dense(10)

    def forward(self, x):
        self.hidden1 = nd.relu(self.dense0(x))
        self.hidden2 = nd.relu(self.dense1(self.hidden1))
        self.hidden3 = nd.relu(self.dense2(self.hidden2))
        x = self.dense3(self.hidden3)
        return x

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

batch_size = 64

net = MLP()
net.load_parameters("../weights/part_a/NN2.params")


if mode == 'train':

    images_train , labels_train = DataLoader().load_data("train")
    X_train, X_val, y_train, y_val = train_test_split(images_train, labels_train, test_size=0.30, random_state=42)

    train_data = []
    for index,data in enumerate(X_train):
        temp = y_train[index]
        train_data.append((data,temp))
        
    val_data = []
    for index,data in enumerate(X_val):
        temp = y_val[index]
        val_data.append((data,temp))
        
    train_data  = mx.gluon.data.DataLoader(train_data, batch_size,shuffle = True)
    val_data = mx.gluon.data.DataLoader(val_data, batch_size,shuffle = False)

    hidden1_train = []
    hidden2_train = []
    hidden3_train = []
    y_train = []

    hidden1_val = []
    hidden2_val = []
    hidden3_val = []
    y_val = []

    for i, (data, label) in enumerate(train_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            hidden3_train.append(net.hidden3.asnumpy())
            hidden2_train.append(net.hidden2.asnumpy())
            hidden1_train.append(net.hidden1.asnumpy())
            y_train.append(label.asnumpy().reshape(-1,1))

    for i, (data, label) in enumerate(val_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            hidden3_val.append(net.hidden3.asnumpy())
            hidden2_val.append(net.hidden2.asnumpy())
            hidden1_val.append(net.hidden1.asnumpy())
            y_val.append(label.asnumpy().reshape(-1,1))

    hidden1_train = np.vstack(hidden1_train)
    hidden1_val = np.vstack(hidden1_val)

    hidden2_train = np.vstack(hidden2_train)
    hidden2_val = np.vstack(hidden2_val)

    hidden3_train = np.vstack(hidden3_train)
    hidden3_val = np.vstack(hidden3_val)

    y_train = np.vstack(y_train).reshape(-1,)
    y_val = np.vstack(y_val).reshape(-1,)

    model_h1 = LogisticRegression(random_state=42)
    model_h1.fit(hidden1_train, y_train)
    print('Train accuracy when taken from hidden 1 is ', model_h1.score(hidden1_train, y_train))
    print('Validation accuracy when taken from hidden 1 is ', model_h1.score(hidden1_val, y_val))
    pkl.dump(model_h1, open('../weights/model_hidden1', 'wb'))

    model_h2 = LogisticRegression(random_state=42)
    model_h2.fit(hidden2_train, y_train)
    print('Train accuracy when taken from hidden 2 is ', model_h2.score(hidden2_train, y_train))
    print('Validation accuracy when taken from hidden 2 is ', model_h2.score(hidden2_val, y_val))
    pkl.dump(model_h2, open('../weights/model_hidden2', 'wb'))

    model_h3 = LogisticRegression(random_state=42)
    model_h3.fit(hidden3_train, y_train)
    print('Train accuracy when taken from hidden 3 is ', model_h3.score(hidden3_train, y_train))
    print('Validation accuracy when taken from hidden 3 is ', model_h3.score(hidden3_val, y_val))
    pkl.dump(model_h3, open('../weights/model_hidden3', 'wb'))

if mode == 'test':

    images_test , labels_test = DataLoader().load_data("test")  

    test_data = []
    for index,data in enumerate(images_test):
        temp = labels_test[index]
        test_data.append((data,temp))

    test_data = mx.gluon.data.DataLoader(test_data, batch_size,shuffle = False)

    hidden1_test = []
    hidden2_test = []
    hidden3_test = []
    y_test = []

    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            hidden3_test.append(net.hidden3.asnumpy())
            hidden2_test.append(net.hidden2.asnumpy())
            hidden1_test.append(net.hidden1.asnumpy())
            y_test.append(label.asnumpy().reshape(-1,1))

    hidden1_test = np.vstack(hidden1_test)
    hidden2_test = np.vstack(hidden2_test)
    hidden3_test = np.vstack(hidden3_test)

    y_test = np.vstack(y_test).reshape(-1,)

    model_h1 = pkl.load(open('../weights/part_c/model_hidden1', 'rb'))
    model_h2 = pkl.load(open('../weights/part_c/model_hidden2', 'rb'))
    model_h3 = pkl.load(open('../weights/part_c/model_hidden3', 'rb'))

    print('Accuracy for Hidden layer 1 model:', model_h1.score(hidden1_test, y_test))
    print('Accuracy for Hidden layer 2 model:', model_h2.score(hidden2_test, y_test))
    print('Accuracy for Hidden layer 3 model:', model_h3.score(hidden3_test, y_test))

