import json
import numpy as np
import random

def get_vocab_size():
	'''open json file word count and return vocabulary size'''
	with open('./word_count.json','r') as f:
		word_count = json.loads(f.read())
	vocab_size = len(word_count['word_count'])
	return vocab_size

class DataReader:
	'''Read the file through each batch'''
	def __init__(self,path_file,batch_size):
		self._path_file = path_file
		self._batch_size = batch_size
		self._data = []
		self._label = []
		self._sentence_length =[]
		self._id = 0
		
		'''open data encoded and get label, sentence length, data'''
		with open(path_file,'r') as f:
			document = f.read()
			document = document.split('\n')
			for line in document:
				features = line.split('<fff>')
				self._label.append(int(features[0]))
				self._sentence_length.append(int(features[2]))
				data = features[3].split()
				data_ = [int(_) for _ in data]
				self._data.append(data_)
		
		'''number batch'''
		self._num_step = int(len(self._data)/self._batch_size)
		
		'''shuffle data'''
		indices = [index for index in range(len(self._data))]
		random.seed(1000)
		random.shuffle(indices)
		
		self._data = np.array(self._data)
		self._label = np.array(self._label)
		self._sentence_length = np.array(self._sentence_length)
		
		self._data = self._data[indices]
		self._label = self._label[indices]
		self._sentence_length = self._sentence_length[indices]
		
	def next_batch(self):
		'''get batch data'''
		if (self._id < self._num_step):
			start = self._id*self._batch_size
			end = (self._id+1)*self._batch_size

			batch_data = self._data[start:end]
			batch_label = self._label[start:end]
			batch_sentence_length = self._sentence_length[start:end]

			self._id += 1
		# if lenth batch last < batch size, start batch last = length data - batch size
		else:
			start = len(self._data) - self._batch_size
			batch_data = self._data[start:]
			batch_label = self._label[start:]
			batch_sentence_length = self._sentence_length[start:]

			self._id = 0

		return (batch_data,batch_label,batch_sentence_length)
		 
if __name__ == '__main__':
	print(get_vocab_size())
