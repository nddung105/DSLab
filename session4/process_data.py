import os
import re
from collections import defaultdict 
import json
	
def get_vocab():
	'''get vocabulary, write to file json'''
	path_train = "../session1/data/20news-bydate-train"
	#path_test = "../session1/data/20news-bydate-test"
	list_train = os.listdir(path_train)
	#list_test = os.listdir(path_test)
	
	'''list path all folder in train data'''
	list_path_folder = []
	for folder in list_train:
		list_path_folder.append(os.path.join(path_train,folder))
		#list_path_folder.append(os.path.join(path_test,folder))
	
	
	word_count = defaultdict(lambda: {'count':0,'id':0})
	'''loop through each folder'''
	for path_folder in list_path_folder:
		list_path_file = [os.path.join(path_folder,file_name) for file_name in os.listdir(path_folder)]
		'''loop through each file'''
		for path_file in list_path_file:
			print(path_file)
			with open(path_file, 'r',encoding="utf-8",errors='ignore') as f:
				data = f.read()
			'''remove the numbers and split'''
			data = re.sub(r'[0-9]', '', data)
			words = re.split('\W+', data)
			for word in words:
				word_count[word]['count'] += 1
	index = 0
	word_count_ = defaultdict(lambda: {'count':0,'id':0})
	'''add count word and id'''
	for key,value in word_count.items():
		if (value['count'] > 10):
			word_count_[key]= word_count[key]
			word_count_[key]['id'] = index
			index += 1
		
	json_word_count = {
						'word_count' : word_count_
						}
	'''write json file'''
	with open('./word_count.json','w',encoding='utf8') as f:
		json.dump(json_word_count,f,ensure_ascii=False)

	return 0
	
def encode_data(path_file,max_length = 500):
	'''write text and traing data encoded'''
	
	unknown_ID = 1
	padding_ID = 0
	
	with open('./word_count.json','r') as f:
		word_count = json.loads(f.read())
	with open(path_file,'r') as f:
		data = f.read()
		
	encode_data = []
	'''loop through each document'''
	for document in data.split('\n'):
		'''get label, doc id, text'''
		(label,doc_id,text) = document.split('<fff>')
		text = re.sub(r'[0-9]','',text)
		words = text.split()[:max_length]
		sentence_length = len(words)
		encode_text = []
		
		'''encode data, add padding_id and unknown_id'''
		if len(words) >= max_length:
			for word in words[:max_length]:
				if (word in word_count['word_count'].keys()):
					encode_text.append(str(word_count['word_count'][word]['id']+2))
				else:
					encode_text.append(str(unknown_ID))
		else:
			num_padding = max_length - len(words)
			for word in words:
				if (word in word_count['word_count'].keys()):
					encode_text.append(str(word_count['word_count'][word]['id']+2))
				else:
					encode_text.append(str(unknown_ID))
			for _ in range(num_padding):
				encode_text.append(str(padding_ID))
		line = label + '<fff>' + doc_id + '<fff>' +str(sentence_length)+'<fff>'+ ' '.join(encode_text) 
		encode_data.append(line)
	
	'''save data encoded'''
	data = '\n'.join(encode_data)
	name_file_new = path_file.split('/')[-1].replace('.txt','-encode.txt')
	print(name_file_new)
	path_file_new = './data_encode/' + name_file_new
	with open(path_file_new,'w') as f:
		f.write(data)
				
if __name__ == "__main__":
	#get_vocab()
	encode_data('../session1/data/20news-train-processed.txt')
	#encode_data('../session1/data/20news-test-processed.txt')
