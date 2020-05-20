import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression
print(tf.__version__)


class MLP:
    """docstring for MLP"""

    def __init__(self, vocb_size, hidden_unit, number_classer):
        super(MLP, self).__init__()
        self.vocb_size = vocb_size  # number of words in vocabulary
        self.hidden_unit = hidden_unit  # number unit of hidden layer
        self.NUMBER_CLASSES = number_classer  # number class

    # Build model and return loss function, label predict
    def build_model(self):
        # X,Y will be given at run time
        # X shape = (number of examples, vocabulary size)
        # Y shape = (number of examples)
        self.X = tf.placeholder(tf.float32, shape=[None, self.vocb_size])
        self.Y = tf.placeholder(tf.int32, shape=[None, ])
        # weights and bias are variables to be updated
        weight_1 = tf.get_variable(name="weight_input_hidden", shape=(
            self.vocb_size, self.hidden_unit), initializer=tf.random_normal_initializer(1000))
        weight_2 = tf.get_variable(name="weight_hidden_output", shape=(
            self.hidden_unit, self.NUMBER_CLASSES), initializer=tf.random_normal_initializer(1000))
        bias_1 = tf.get_variable(name="bias_input_hidden", shape=(
            self.hidden_unit), initializer=tf.random_normal_initializer(1000))
        bisas_2 = tf.get_variable(
            name="bias_hidden_output", shape=(self.NUMBER_CLASSES), initializer=tf.random_normal_initializer(1000))
        # Calculate value of hidden layer
        hidden_units = tf.sigmoid(tf.matmul(self.X, weight_1) + bias_1)
        # Calculate value of output
        output_units = tf.matmul(hidden_units, weight_2) + bisas_2
        # Convert Y to form one hot
        label_one_hot = tf.one_hot(
            indices=self.Y, depth=self.NUMBER_CLASSES, dtype=tf.float32)
        # Calculate loss with cross entropy
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=label_one_hot, logits=output_units)
        loss = tf.reduce_mean(loss)
        # Predict label from output
        predict_label = tf.squeeze(
            tf.argmax(tf.nn.softmax(output_units), axis=1))
        return predict_label, loss

    # Set optimal algorithm
    def trainer(self, loss, learn_rate):
        # set Adam optimal with learn rate and minimize loss
        train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        return train_op


class DataReader(object):
    """docstring for DataReader"""
    # Read data, label from file tf-idf. Function next_batch get data each
    # batch

    def __init__(self, path_file, batch_size, vocb_size):
        super(DataReader, self).__init__()
        self.path_file = path_file
        self.vocb_size = vocb_size
        self.batch_size = batch_size
        self.data = []
        self.label = []
        # creat vector with dim = vocabulary size
        vector_line = [0.0 for _ in range(vocb_size)]
        # get label, data from file
        with open(path_file) as f:
            data = f.read().split("\n")
        for line in data:
            feature = line.split("<fff>")
            self.label.append(int(feature[0]))
            for word in feature[2].split():
                vector_line[int(word.split(':')[0])] = float(
                    word.split(':')[1])
            # vector line, value of index is tf-idf value
            self.data.append(vector_line)
        # array data, label
        self.data = np.array(self.data)
        self.label = np.array(self.label)
        # set number of epoch, batch id
        self.num_epoch = 0
        self.batch_id = 0

    def next_batch(self):
        start = self.batch_id * self.batch_size
        end = start + self.batch_size
        # if call next batch, batch id + 1
        self.batch_id += 1
        # if end +batch size >  number of examples => end = number of examples,
        # set batch id = 0, num epoch + 1
        # and shuffle data
        if ((end + self.batch_size) > len(self.data)):
            end = len(self.data)
            self.num_epoch += 1
            self.batch_id = 0
            indices = [index for index in range(len(self.data))]
            random.seed(1000)
            random.shuffle(indices)
            self.data, self.label = self.data[indices], self.label[indices]
        # return data, label have length = batch size
        return self.data[start:end], self.label[start:end]


def train(vocb_size, hidden_unit, number_classer, learn_rate, epoch, path_train, path_test):
    print("------load data train------")
    # load data train
    train_reader = DataReader(path_file=path_train,
                              batch_size=50, vocb_size=vocb_size)
    print("------load data test------")
    # load data test
    test_reader = DataReader(
        path_file=path_test, batch_size=50, vocb_size=vocb_size)
    print("------load finish------")
    # create model MLP
    mlp = MLP(vocb_size=vocb_size, hidden_unit=hidden_unit,
              number_classer=number_classer)
    predict_label, loss = mlp.build_model()
    train_opt = mlp.trainer(loss, learn_rate)
    # index_epoch is list range number epoch, loss_of_epoch is list loss with
    # each epoch
    index_epoch = []
    loss_of_epoch = []
    accuracy_epoch = []
    # create session
    with tf.Session() as sess:
        epoch, MAX_EPOCH = 0, epoch
        sess.run(tf.global_variables_initializer())
        while epoch < MAX_EPOCH:
            loss_print = 0
            if train_reader.batch_id == 0:
                count = 1
            while count != 0:
                # get batch data, label
                train_data, train_label = train_reader.next_batch()
                count = train_reader.batch_id
                # run
                _, loss_, _ = sess.run([predict_label, loss, train_opt], feed_dict={
                    mlp.X: train_data,
                    mlp.Y: train_label})
                loss_print = loss_
            epoch += 1
            print("------epoch {} ------loss {}".format(epoch, loss_print))
            index_epoch.append(epoch)
            loss_of_epoch.append(loss_print)
            # get accuracy of epoch
            num_true_preds = 0
            while True:
                test_data, test_label = test_reader.next_batch()
                test_plabels_eval = sess.run(predict_label,
                                             feed_dict={
                                                 mlp.X: test_data,
                                                 mlp.Y: test_label
                                             })
                matches = np.equal(test_plabels_eval, test_label)
                num_true_preds += np.sum(matches.astype(float))
                if(test_reader.batch_id == 0):
                    break
            acc = num_true_preds / len(test_reader.data)
            accuracy_epoch.append(acc)
            print('------Accuracy ', acc)
        # save paramenters
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            save_paramenters(name=variable.name,
                             value=variable.eval(), epoch=epoch)
    plt.plot(index_epoch, loss_of_epoch)
    plt.plot(index_epoch, accuracy_epoch)
    plt.show()


def save_paramenters(name, value, epoch):
    # save paramenters trained of model
    # name is name of variable
    filename = name.replace(':', '-') + '-epoch-{}.txt'.format(epoch)
    # if length value shape = 1 => bias
    # else => weight
    if(len(value.shape) == 1):
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number) for number in value[
                                row]]) for row in range(value.shape[0])])
    with open('./save_paras/' + filename, 'w') as f:
        f.write(string_form)


def restore_paramenters(name, epoch):
    # restore paramenters trained of model
    # name is name of variable
    filename = name.replace(':', '-') + '-epoch-{}.txt'.format(epoch)
    with open('./save_paras/' + filename, 'r') as f:
        list_line = f.read().split('\n')
    # if number of line = 1 => bias
    # else => weight
    if(len(list_line) == 1):
        value = [float(number) for number in list_line[0].split(',')]
    else:
        value = [[float(number) for number in list_line[index].split(',')]
                 for index in range(len(list_line))]
    return value


def test_model(vocb_size, hidden_unit, number_classer, epoch, path_test):
    # function get accuracy model trained
    mlp = MLP(vocb_size=vocb_size, hidden_unit=hidden_unit,
              number_classer=number_classer)
    predict_label, loss = mlp.build_model()
    # read data test
    test_reader = DataReader(
        path_file=path_test, batch_size=50, vocb_size=vocb_size)
    with tf.Session() as sess:
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_paramenters(variable.name, epoch=epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)
        num_true_preds = 0
        # loop data test and predict
        while True:
            test_data, test_label = test_reader.next_batch()
            test_plabels_eval = sess.run(predict_label,
                                         feed_dict={
                                             mlp.X: test_data,
                                             mlp.Y: test_label
                                         })
            matches = np.equal(test_plabels_eval, test_label)
            # num_true_preds  is number of predict label true
            num_true_preds += np.sum(matches.astype(float))
            if(test_reader.batch_id == 0):
                break
        print('------Accuracy ', num_true_preds / len(test_reader.data))


def logistic_regression():
    # test with logistic regression of Sklearn
    train_reader = DataReader(path_file="../session1/data/data_train_tf_idf.txt",
                              batch_size=50, vocb_size=14140)
    test_reader = DataReader(path_file="../session1/data/data_test_tf_idf.txt",
                             batch_size=50, vocb_size=14140)
    clf = LogisticRegression(random_state=0).fit(
        train_reader.data, train_reader.label)
    print(clf.score(test_reader.data, test_reader.label))


if __name__ == '__main__':

    train(vocb_size=14140, hidden_unit=10, number_classer=20, learn_rate=0.001, epoch=5,
          path_train="../session1/data/data_train_tf_idf.txt", path_test="../session1/data/data_train_tf_idf.txt")
    test_model(vocb_size=14140, hidden_unit=100, number_classer=20,
               epoch=10, path_test="../session1/data/data_train_tf_idf.txt")
    logistic_regression()
