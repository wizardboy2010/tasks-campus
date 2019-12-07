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

class MLP(gluon.Block):
    def __init__(self, dropout = 0, batch_norm = False, **kwargs):
        self.batch_norm = batch_norm
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(1024)
            self.drop0 = gluon.nn.Dropout(dropout)
            self.dense1 = gluon.nn.Dense(512)
            self.drop1 = gluon.nn.Dropout(dropout)
            self.dense2 = gluon.nn.Dense(256)
            self.drop2 = gluon.nn.Dropout(dropout)
            self.dense3 = gluon.nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = self.drop0(x)
        if self.batch_norm:
            x = self.norm1(x)
        x = nd.relu(self.dense1(x))
        x = self.drop1(x)
        if self.batch_norm:
            x = self.norm2(x)
        x = nd.relu(self.dense2(x))
        x = self.drop2(x)
        if self.batch_norm:
            x = self.norm3(x)
        x = self.dense3(x)
        return x

class MLP_batch(gluon.Block):
    def __init__(self, dropout = 0, batch_norm = False, **kwargs):
        self.batch_norm = batch_norm
        super(MLP_batch, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(1024)
            self.drop0 = gluon.nn.Dropout(dropout)
            self.norm1 = gluon.nn.BatchNorm()
            self.dense1 = gluon.nn.Dense(512)
            self.drop1 = gluon.nn.Dropout(dropout)
            self.norm2 = gluon.nn.BatchNorm()
            self.dense2 = gluon.nn.Dense(256)
            self.drop2 = gluon.nn.Dropout(dropout)
            self.norm3 = gluon.nn.BatchNorm()
            self.dense3 = gluon.nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = self.drop0(x)
        if self.batch_norm:
            x = self.norm1(x)
        x = nd.relu(self.dense1(x))
        x = self.drop1(x)
        if self.batch_norm:
            x = self.norm2(x)
        x = nd.relu(self.dense2(x))
        x = self.drop2(x)
        if self.batch_norm:
            x = self.norm3(x)
        x = self.dense3(x)
        return x

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

batch_size = 32
epochs = 30

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

    ############################################################ vanilla model

    net = MLP()
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    vanilla_train_loss = []

    print('training of Vanilla Model')

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
        vanilla_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/vanilla.params")
    # np.save('vanilla', vanilla_train_loss)

    ####################################################  Experiement 1

    ####### Normal init

    net = MLP()
    net.collect_params().initialize(mx.init.Normal(), ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    exp_1_normal_train_loss = []

    print('training of Normal initialization Model')

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
        exp_1_normal_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_1_normal.params")
    # np.save('Normal', exp_1_normal_train_loss)

    ####### xavier init

    net = MLP()
    net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    exp_1_xavier_train_loss = []

    print('training of Xavier initialization Model')

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
        exp_1_xavier_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_1_xavier.params")
    # np.save('xavier', exp_1_xavier_train_loss)

    ####### orthogonal init

    net = MLP()
    net.collect_params().initialize(mx.init.Orthogonal(), ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    exp_1_ortho_train_loss = []

    print('training of Orthogonal initialization Model')

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
        exp_1_ortho_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_1_ortho.params")
    # np.save('ortho', exp_1_ortho_train_loss)

    ############# exp 1 plot

    plt.figure("Image")
    plt.title("Experiement 1 - Initialization")
    vanilla_train_loss =  [float(i)/sum(vanilla_train_loss) for i in vanilla_train_loss]
    exp_1_normal_train_loss =  [float(i)/sum(exp_1_normal_train_loss) for i in exp_1_normal_train_loss]
    exp_1_xavier_train_loss =  [float(i)/sum(exp_1_xavier_train_loss) for i in exp_1_xavier_train_loss]
    exp_1_ortho_train_loss =  [float(i)/sum(exp_1_ortho_train_loss) for i in exp_1_ortho_train_loss]
    plt.plot(vanilla_train_loss, label="Vanilla train Loss")
    plt.plot(exp_1_normal_train_loss, label="Normal init train Loss")
    plt.plot(exp_1_xavier_train_loss, label="Xavier init traib Loss")
    plt.plot(exp_1_ortho_train_loss, label = "Orthogonal init train Loss")
    plt.legend()
    plt.savefig('../report/part_b_exp_1.png')
    #plt.show()

    #################################################### Experiment 2

    net = MLP_batch(batch_norm = True)
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    exp_2_batchnorm_train_loss = []

    print('training of BatchNorm Model')

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
        exp_2_batchnorm_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_2_batchnorm.params")
    # np.save('batch_norm', exp_2_batchnorm_train_loss)

    ############# exp 2 plot

    #vanilla_train_loss = np.load('vanilla.npy')

    plt.figure("Image")
    plt.title("Experiement 2 - Barch Normalization")
    vanilla_train_loss =  [float(i)/sum(vanilla_train_loss) for i in vanilla_train_loss]
    exp_2_batchnorm_train_loss =  [float(i)/sum(exp_2_batchnorm_train_loss) for i in exp_2_batchnorm_train_loss]
    plt.plot(vanilla_train_loss, label="Vanilla train Loss")
    plt.plot(exp_2_batchnorm_train_loss, label="Batch Normalization train Loss")
    plt.legend()
    plt.savefig('../report/part_b_exp_2.png')
    #plt.show()

    ####################################################  Experiement 3

    ##### drop out = 0.1

    net = MLP(dropout = 0.1)
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    drop10_train_loss = []

    print('training of Dropout 0.1 Model')

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
        drop10_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_3_drop_10.params")
    # np.save('drop10', drop10_train_loss)

    ##### drop out = 0.4

    net = MLP(dropout = 0.4)
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    drop40_train_loss = []

    print('training of Dropout 0.4 Model')

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
        drop40_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_3_drop_40.params")
    # np.save('drop40', drop40_train_loss)

    ##### drop out = 0.6

    net = MLP(dropout = 0.6)
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    drop60_train_loss = []

    print('training of Dropout 0.6 Model')

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
        drop60_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_3_drop_60.params")
    # np.save('drop60', drop60_train_loss)

    ############# exp 3 plot

    plt.figure("Image")
    plt.title("Experiement 3 - Regularization(Dropout)")
    vanilla_train_loss =  [float(i)/sum(vanilla_train_loss) for i in vanilla_train_loss]
    drop10_train_loss =  [float(i)/sum(drop10_train_loss) for i in drop10_train_loss]
    drop40_train_loss =  [float(i)/sum(drop40_train_loss) for i in drop40_train_loss]
    drop60_train_loss =  [float(i)/sum(drop60_train_loss) for i in drop60_train_loss]
    plt.plot(vanilla_train_loss, label="Vanilla train Loss")
    plt.plot(drop10_train_loss, label="Dropout 0.1 train Loss")
    plt.plot(drop40_train_loss, label="Dropout 0.4 traib Loss")
    plt.plot(drop60_train_loss, label = "Dropout 0.6 train Loss")
    plt.legend()
    plt.savefig('../report/part_b_exp_3.png')
    #plt.show()

    ####################################################  Experiement 4

    ##### SGD

    net = MLP()
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 1e-3})

    exp_4_sgd_train_loss = []

    print('training of sgd Model')

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
        exp_4_sgd_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_4_sgd.params")
    # np.save('sgd', exp_4_sgd_train_loss)

    ##### NAG

    net = MLP()
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'nag', {'learning_rate': 1e-3})

    exp_4_nag_train_loss = []

    print('training of NAG Model')

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
        exp_4_nag_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_4_nag.params")
    # np.save('nag', exp_4_nag_train_loss)

    ##### AdaDelta

    net = MLP()
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adadelta', {'learning_rate': 1e-3})

    exp_4_adadelta_train_loss = []

    print('training of AdaDelta Model')

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
        exp_4_adadelta_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_4_adadelta.params")
    # np.save('adadelta', exp_4_adadelta_train_loss)

    ##### adagrad

    net = MLP()
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adagrad', {'learning_rate': 1e-3})

    exp_4_adagrad_train_loss = []

    print('training of adagrad Model')

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
        exp_4_adagrad_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_4_adagrad.params")
    # np.save('adagrad', exp_4_adagrad_train_loss)

    ##### rmsprop

    net = MLP()
    net.collect_params().initialize(ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'rmsprop', {'learning_rate': 1e-3})

    exp_4_rmsprop_train_loss = []

    print('training of rmsprop Model')

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
        exp_4_rmsprop_train_loss.append(cumulative_loss/num_examples)
        print("Epoch %s. Loss: %s, Train_acc %s, Valid_acc %s, Valid_Loss %s, Time For Epoch(in secs) %s" %
              (e+1, cumulative_loss/num_examples, train_accuracy, val_accuracy, v_loss/num_examples, difference))

    net.save_parameters("../weights/part_b/exp_4_rmsprop.params")
    # np.save('rmsprop', exp_4_rmsprop_train_loss)

    ############# exp 4 plot

    plt.figure("Image")
    plt.title("Experiement 4 - Optimizers")
    vanilla_train_loss =  [float(i)/sum(vanilla_train_loss) for i in vanilla_train_loss]
    exp_4_sgd_train_loss =  [float(i)/sum(exp_4_sgd_train_loss) for i in exp_4_sgd_train_loss]
    exp_4_nag_train_loss =  [float(i)/sum(exp_4_nag_train_loss) for i in exp_4_nag_train_loss]
    exp_4_adadelta_train_loss =  [float(i)/sum(exp_4_adadelta_train_loss) for i in exp_4_adadelta_train_loss]
    exp_4_adagrad_train_loss =  [float(i)/sum(exp_4_adagrad_train_loss) for i in exp_4_adagrad_train_loss]
    exp_4_rmsprop_train_loss =  [float(i)/sum(exp_4_rmsprop_train_loss) for i in exp_4_rmsprop_train_loss]
    plt.plot(vanilla_train_loss, label="Adam train Loss(vanilla)")
    plt.plot(exp_4_sgd_train_loss, label="SGD train Loss")
    plt.plot(exp_4_nag_train_loss, label="NAG traib Loss")
    plt.plot(exp_4_adadelta_train_loss, label="AdaDelta traib Loss")
    plt.plot(exp_4_adagrad_train_loss, label="adagrad traib Loss")
    plt.plot(exp_4_rmsprop_train_loss, label = "rmsprop train Loss")
    plt.legend()
    plt.savefig('../report/part_b_exp_4.png')
    #plt.show()

if mode == 'test':
    images_test , labels_test = DataLoader().load_data("test")  

    test_data = []
    for index,data in enumerate(images_test):
        temp = labels_test[index]
        test_data.append((data,temp))

    test_data = mx.gluon.data.DataLoader(test_data, batch_size,shuffle = False)

    net = MLP()
    net.load_parameters("../weights/part_b/vanilla.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Vanilla Model: ", float(accuracy/cnt))

    print("\nExperiment 1 - Initialization\n")

    net = MLP()
    net.load_parameters("../weights/part_b/exp_1_normal.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Normal Initialization Model: ", float(accuracy/cnt))

    net = MLP()
    net.load_parameters("../weights/part_b/exp_1_xavier.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Xavier Initialization Model: ", float(accuracy/cnt))

    net = MLP()
    net.load_parameters("../weights/part_b/exp_1_ortho.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Orthogonal Initialization Model: ", float(accuracy/cnt))

    print("\nExperiment 2 - Batch Normalization\n")

    net = MLP_batch()
    net.load_parameters("../weights/part_b/exp_2_batchnorm.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Batch Normalization Model: ", float(accuracy/cnt))

    print("\nExperiment 3 - Regularization(Dropout)\n")

    net = MLP()
    net.load_parameters("../weights/part_b/exp_3_drop_10.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Dropout 0.1 Model: ", float(accuracy/cnt))

    net = MLP()
    net.load_parameters("../weights/part_b/exp_3_drop_40.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Dropout 0.4 Model: ", float(accuracy/cnt))

    net = MLP()
    net.load_parameters("../weights/part_b/exp_3_drop_60.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for Dropout 0.6 Model: ", float(accuracy/cnt))

    print("\nExperiment 4 - Optimizers\n")

    net = MLP()
    net.load_parameters("../weights/part_b/exp_4_sgd.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for SGD Optimizer Model: ", float(accuracy/cnt))

    net = MLP()
    net.load_parameters("../weights/part_b/exp_4_nag.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for NAG Optimizer Model: ", float(accuracy/cnt))

    net = MLP()
    net.load_parameters("../weights/part_b/exp_4_adadelta.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for AdaDelta Optimizer Model: ", float(accuracy/cnt))

    net = MLP()
    net.load_parameters("../weights/part_b/exp_4_adagrad.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for AdaGrad Optimizer Model: ", float(accuracy/cnt))

    net = MLP()
    net.load_parameters("../weights/part_b/exp_4_rmsprop.params")
    #print(net.collect_params())

    cnt = 0
    accuracy = 0
    for i, (data, label) in enumerate(test_data):
        data , label = transform(data,label)
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1

    print("Total Accuracy for RmsProp Optimizer Model: ", float(accuracy/cnt))
