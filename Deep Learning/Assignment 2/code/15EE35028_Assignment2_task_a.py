from zipfile import ZipFile
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import sys

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

class MLP_network_1(gluon.Block):
    def __init__(self, **kwargs):
        super(MLP_network_1, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(512)
            self.dense1 = gluon.nn.Dense(128)
            self.dense2 = gluon.nn.Dense(64)
            self.dense3 = gluon.nn.Dense(32)
            self.dense4 = gluon.nn.Dense(16)
            self.dense5 = gluon.nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = nd.relu(self.dense1(x))
        x = nd.relu(self.dense2(x))
        x = nd.relu(self.dense3(x))
        x = nd.relu(self.dense4(x))
        x = self.dense5(x)
        return x

class MLP_network_2(gluon.Block):
    def __init__(self, **kwargs):
        super(MLP_network_2, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(1024)
            self.dense1 = gluon.nn.Dense(512)
            self.dense2 = gluon.nn.Dense(256)
            self.dense3 = gluon.nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = nd.relu(self.dense1(x))
        x = nd.relu(self.dense2(x))
        x = self.dense3(x)
        return x

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

batch_size = 64


if mode == 'train':
    images_train , labels_train = DataLoader().load_data("train")
    X_train, X_val, y_train, y_val = train_test_split(images_train, labels_train, test_size=0.30, random_state=42)

    num_inputs = 784
    num_outputs = 10
    num_examples = 60000

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

    ###################### Model 1
    net = MLP_network_1()
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})
    epochs = 20

    network_1_train_loss = []
    network_1_valid_loss = []

    for e in range(epochs):
        cumulative_loss = 0
        now = datetime.now()
        for i, (data, label) in enumerate(train_data):
            data , label = transform(data,label)
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)

            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy, v_loss = evaluate_accuracy(val_data, net)
        train_accuracy = evaluate_accuracy_train(train_data, net)
        later = datetime.now()
        difference = (later-now).total_seconds()
        network_1_train_loss.append(cumulative_loss/num_examples)
        network_1_valid_loss.append(v_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))


    plt.figure("Image")
    plt.title("Network 1 Loss vs Epoch")
    network_1_valid_loss =  [float(i)/sum(network_1_valid_loss) for i in network_1_valid_loss]
    network_1_train_loss =  [float(i)/sum(network_1_train_loss) for i in network_1_train_loss]
    plt.plot(network_1_valid_loss, c="red", label="Validation Loss")
    plt.plot(network_1_train_loss, c="blue", label = "Training Loss")
    plt.legend()
    plt.savefig('../report/Network_1_exp_'+str("Image")+'.png')
    plt.show()

    net.save_parameters("../weights/part_a/NN1.params")
    print('params saved')

    ################### model 2
    net = MLP_network_2()
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})
    epochs = 20

    network_2_train_loss = []
    network_2_valid_loss = []

    for e in range(epochs):
        cumulative_loss = 0
        now = datetime.now()
        for i, (data, label) in enumerate(train_data):
            data , label = transform(data,label)
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)

            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        val_accuracy, v_loss = evaluate_accuracy(val_data, net)
        train_accuracy = evaluate_accuracy_train(train_data, net)
        later = datetime.now()
        difference = (later-now).total_seconds()
        network_2_train_loss.append(cumulative_loss/num_examples)
        network_2_valid_loss.append(v_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))


    plt.figure("Image")
    plt.title("Network 2 Loss vs Epoch")
    network_2_valid_loss =  [float(i)/sum(network_2_valid_loss) for i in network_2_valid_loss]
    network_2_train_loss =  [float(i)/sum(network_2_train_loss) for i in network_2_train_loss]
    plt.plot(network_2_valid_loss, c="red", label="Validation Loss")
    plt.plot(network_2_train_loss, c="blue", label = "Training Loss")
    plt.legend()
    plt.savefig('../report/Network_2_exp_'+str("Image")+'.png')
    plt.show()

    net.save_parameters("../weights/part_a/NN2.params")
    print('params saved')

if mode == 'test':
    images_test , labels_test = DataLoader().load_data("test")  

    test_data = []
    for index,data in enumerate(images_test):
        temp = labels_test[index]
        test_data.append((data,temp))

    test_data = mx.gluon.data.DataLoader(test_data, batch_size,shuffle = False)

    net1 = MLP_network_1()
    net1.load_parameters("../weights/part_a/NN1.params")
    #print(net1.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net1(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Network 1: ", float(accuracy/cnt))

    net2 = MLP_network_2()
    net2.load_parameters("../weights/part_a/NN2.params")
    #print(net2.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net2(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Network 2: ", float(accuracy/cnt))
