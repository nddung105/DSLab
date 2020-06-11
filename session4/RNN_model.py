import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from read_data import get_vocab_size, DataReader

MAX_LENGTH = 500
NUM_CLASSES = 20

class RNN:
	def __init__ (self,vocab_size,lstm_size,embedding_size,batch_size):
		self._vocab_size = vocab_size
		self._lstm_size = lstm_size
		self._embedding_size = embedding_size
		self._batch_size = batch_size
		
		'''initialize placeholder data, label, sentence length, final tokens'''
		self._data = tf.placeholder(tf.int32,shape = [batch_size,MAX_LENGTH]) 
		self._label = tf.placeholder(tf.int32,shape = [batch_size,])
		self._sentence_lengths = tf.placeholder(tf.int32,shape = [batch_size,])
		self._final_tokens = tf.placeholder(tf.int32,shape = [batch_size,])
	
	def embedding_layer(self,indices):
		'''layer embedding'''
		
		pretrained_vectors = [np.zeros(self._embedding_size)]
		np.random.seed(1000)
		for _ in range(self._vocab_size + 1):
			pretrained_vectors.append(np.random.normal(loc = 0., scale = 1.,size = self._embedding_size ))
		pretrained_vectors = np.array(pretrained_vectors)
		self._embedding_matrix = tf.get_variable(
			name = 'embedding',
			shape = (self._vocab_size + 2,self._embedding_size),
			initializer = tf.constant_initializer(pretrained_vectors))
		return tf.nn.embedding_lookup(self._embedding_matrix,indices)
	
	def LSTM_layer(self,embedding):
		'''lstm layer'''
		
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
		zero_state = tf.zeros(shape = (self._batch_size,self._lstm_size))
		initial_state = tf.contrib.rnn.LSTMStateTuple(zero_state,zero_state)
		lstm_inputs = tf.unstack(tf.transpose(embedding,perm = [1, 0, 2]))
		
		'''a length-500 list of [num_docs, lstm_size]'''
		lstm_outputs, last_state = tf.nn.static_rnn(
			cell = lstm_cell,
			inputs = lstm_inputs,
			initial_state = initial_state,
			sequence_length = self._sentence_lengths)
			
		lstm_outputs = tf.unstack(tf.transpose(lstm_outputs,perm=[1,0,2]))
		
		'''[num docs * MAX_SENT_LENGTH, lstm_size]'''
		lstm_outputs = tf.concat(lstm_outputs,axis = 0)
		
		'''[num docs * MAX_SENT_LENGTH, ]'''
		mask = tf.sequence_mask(
			lengths=self._sentence_lengths,
			maxlen=MAX_LENGTH,
			dtype=tf.float32
		)
		mask = tf.concat(tf.unstack(mask, axis=0), axis=0)
		mask = tf.expand_dims(mask, -1)
		lstm_outputs = mask * lstm_outputs
		
		lstm_outputs_split = tf.split(lstm_outputs, num_or_size_splits=self._batch_size)
		lstm_outputs_sum = tf.reduce_sum(lstm_outputs_split, axis=1)
		
		lstm_outputs_average = lstm_outputs_sum / tf.expand_dims(
			tf.cast(self._sentence_lengths, tf.float32), -1
		)
		return lstm_outputs_average

	def model(self):
		'''build model'''
		
		'''embedding layer'''
		embedding = self.embedding_layer(self._data)
		
		'''lstm layer'''
		lstm_outputs = self.LSTM_layer(embedding)
		
		'''one hidden layer for classification'''
		weights = tf.get_variable(
			name = 'final_layer_weights',
			shape = (self._lstm_size,NUM_CLASSES),
			initializer = tf.random_normal_initializer(seed = 1000))
		biases = tf.get_variable(
			name = 'final_layer_biases',
			shape = (NUM_CLASSES),
			initializer = tf.random_normal_initializer(seed = 1000))
		logits = tf.matmul(lstm_outputs, weights) + biases
		labels_one_hot = tf.one_hot(
			indices = self._label,
			depth = NUM_CLASSES,
			dtype = tf.float32)
		loss = tf.nn.softmax_cross_entropy_with_logits(
			labels = labels_one_hot,
			logits = logits)
		loss = tf.reduce_mean(loss)
		probs = tf.nn.softmax(logits)
		predicted_labels = tf.argmax(probs, axis=1)
		predicted_labels = tf.squeeze(predicted_labels)

		return predicted_labels, loss
	
	def trainer(self,loss,lr=0.001):
	
		'''train with learning rate Adam optimizer'''
		train_op = tf.train.AdamOptimizer(lr).minimize(loss)
		return train_op

def training_model(lstm_size,embedding_size,batch_size, learning_rate):

	'''build model'''
	vocab_size = get_vocab_size()
	print(vocab_size)
	rnn_model = RNN(vocab_size,lstm_size,embedding_size,batch_size)
	predicted_labels, loss = rnn_model.model()
	train_op = rnn_model.trainer(loss,lr=learning_rate)
	
	'''read data'''
	print('----------read data -----------')
	train_data_reader = DataReader('./data_encode/20news-test-processed-encode.txt',batch_size=batch_size)
	test_data_reader = DataReader('./data_encode/20news-train-processed-encode.txt',batch_size=batch_size)
	print('----------success ------------')
	
	list_loss = []
	list_acc = []
	
	'''run session'''
	with tf.Session() as sess:
		'''initializer variables'''
		sess.run(tf.global_variables_initializer())
		epoch = 0
		print('---------training -----------')
		
		'''loop training epoch'''
		while epoch < 5:
			for step in range(train_data_reader._num_step+1):
				print('epoch {} {} %'.format(epoch,step*100/train_data_reader._num_step),end="\r", flush=True)
				'''get batch train'''
				train_data, train_label, train_sentence_length = train_data_reader.next_batch()
				label_predict, loss_, _ = sess.run(
					[predicted_labels,loss,train_op],
					feed_dict = {
						rnn_model._data : train_data,
						rnn_model._label : train_label,
						rnn_model._sentence_lengths : train_sentence_length
						})

			list_loss.append(loss_)
			print('epoch {} {}'.format(epoch,loss_))
			epoch += 1
			
			'''caculate accurate test data'''
			acc = 0
			for step_test in range(test_data_reader._num_step + 1):
				test_data, test_label, test_sentence_length = test_data_reader.next_batch()
				label_predict = sess.run(
					predicted_labels,
					feed_dict = {
						rnn_model._data: test_data,
						rnn_model._label: test_label,
						rnn_model._sentence_lengths: test_sentence_length})
				matches = np.equal(label_predict, test_label)
				score = np.sum(matches.astype(float)) 
				acc = acc + score 
			acc = acc / len(test_data_reader._data)
			print('acc {}'.format(acc))
			list_acc.append(acc)

	plt.plot(list_loss,'r-')
	plt.plot(list_acc,'g-')
	plt.show()
	
if __name__ == '__main__':
	training_model(lstm_size = 250,embedding_size = 300,batch_size = 50,learning_rate = 0.001)
