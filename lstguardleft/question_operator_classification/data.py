import os.path
import collections
import pickle
import array

import tensorflow as tf
import numpy as np
import pandas as pd

from config import create_hparams
from collections import defaultdict
from konlpy.tag import Twitter

# global variable 사용 ??
data_x = []
data_y = []
twitter = Twitter()

class DataProcessor:
	def __init__(self, config):

		self.config = config
		self.question_dict = self.load_pickle(self.config.question_pickle_path)
		self.operator_dict = self.load_pickle(self.config.operator_pickle_path)

		self.data_x, self.data_y = self.data_preprocessing()
		self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = self.seperate_dataset()
		
	def data_preprocessing(self):
		# TODO 
		# - 

		# step 1 
		# 원본 데이터 load 후 list 형태로 변환
		df_sop = pd.read_csv(self.config.train_file_path)

		for idx, value in enumerate(df_sop['question']):
			data_x.append([self.question_dict[question] for question in twitter.morphs(value)]) 

		for idx, value in enumerate(df_sop['operator']):
			data_y.append(self.operator_dict[value])	
		# step 2 
		# left Zero-padding, 단 Zero-Padding시에도 dynamic_rnn 사용함.
		# Many to One 모델 적용으로 left Zero-padding
		max_len = self.config.max_context_len

		for idx, val in enumerate(data_x):
			if len(data_x[idx]) < max_len:
				for ind in range(0, max_len - len(data_x[idx])):
					# data_x[idx].append(0)  # right zero-padding
					data_x[idx].insert(0, 0) # left zero-padding
		return data_x, data_y

	def seperate_dataset(self):
		split_train_frac = 0.8
		split_index = int(split_train_frac * len(self.data_x))
		train_x, val_x = np.array(self.data_x[:split_index]), np.array(self.data_x[split_index:]) 
		train_y, val_y = np.array(self.data_y[:split_index]), np.array(self.data_y[split_index:])

		split_test_frac = 0.5
		split_index = int(split_test_frac * len(val_x))

		val_x, test_x = np.array(val_x[:split_index]), np.array(val_x[split_index:])
		val_y, test_y = np.array(val_y[:split_index]), np.array(val_y[split_index:])

		return train_x, train_y, val_x, val_y, test_x, test_y

	def create_question_vocab(self, filename):
		vocab_list = []
		df_sop = pd.read_csv(filename)

		for loop in range(0, df_sop.shape[0]):
			# pos = twitter.pos(df_sop['question'][loop])
			vocab_list = vocab_list + twitter.morphs(df_sop['question'][loop])

		# Counter를 통해서 반복된 형태소 Grouping 
		vocab_list = collections.Counter(vocab_list)
		vocabulary = [x[0] for x in vocab_list.most_common()]
		vocabulary = list(sorted(vocabulary))

		vocab_dic ={w:i for i, w in enumerate(vocabulary)}

		with open(vocab_file,'w') as f:
			for i, w in enumerate(vocab_dic):
				f.write(w + ':' + str(i) + '\n')

		return vocab_dic

	def create_operator_vocab(self, train_file, vocab_file):
		vocab_list = []
		df_sop = pd.read_csv(train_file)

		for loop in range(0, df_sop.shape[0]):
			# pos = twitter.pos(df_sop['operator'][loop])
			vocab_list.append(df_sop['operator'][loop])

		vocab_list = collections.Counter(vocab_list)
		vocabulary = [x[0] for x in vocab_list.most_common()]
		vocabulary = list(sorted(vocabulary))

		vocab_dic ={w:i for i, w in enumerate(vocabulary)}

		with open(vocab_file, 'w') as f:
			for i, w in enumerate(vocab_dic):
				f.write(w + ':' + str(i) + '\n')

		return vocab_dic

	def create_question_vocab_pickle(self, vocab_dict):
		with open(self.config.question_pickle_path, 'wb') as f:
			pickle.dump(vocab_dict, f)

	def create_operator_vocab_pickle(self, operator_dict):
		with open(self.config.operator_pickle_path, 'wb') as f:
			pickle.dump(operator_dict, f)

	def load_pickle(self, vocab_pickle_full_path):
		with open(vocab_pickle_full_path, 'rb') as f:
			return pickle.load(f)

	def load_question_vocab(self, filename):
		vocab = None
		with open(filename) as f:
			vocab = f.read().splitlines()
			dct = defaultdict(int)
			for idx, word in enumerate(vocab):
				dct[word.split(':')[0]] = idx
		return dct

	def load_operator_vocab(self, filename):
		vocab = None
		with open(filename) as f:
			vocab = f.read().splitlines()
			dct = defaultdict(int)
			for idx, word in enumerate(vocab):
				dct[word.split(':')[0]] = idx
		return dct

	#def get_batches(self):


def main(_):

	hp = create_hparams()
	dp = DataProcessor(hp)

	if os.path.isfile(hp.question_vocab_path) == False:
		print('■ step 1. creating question vocabulary file .... ')
		vocab_dict = dp.create_question_vocab(hp.train_file_path, hp.question_vocab_path)
	else:
		print('■ step 1. question vocabulary file exists .... ')
		#vocab_dict = dp.load_question_vocab(hp.question_vocab_path)

	if os.path.isfile(hp.operator_vocab_path) == False:
		print('■ step 2. creating operator vocabulary file .... ')
		operator_dict = dp.create_operator_vocab(hp.train_file_path, hp.operator_vocab_path)
	else:
		print('■ step 2. operator vocabulary file exists .... ')
		#operator_dict = dp.load_operator_vocab(hp.operator_vocab_path)

	if os.path.isfile(hp.question_pickle_path) == False:
		print('■ step 3. creating question pickle file .... ')
		dp.create_question_vocab_pickle(self.vocab_dict)
	else:
		print('■ step 3. question pickle file exists .... ')
		#vocab_dict = dp.load_pickle(hp.question_pickle_path)

	if os.path.isfile(hp.operator_pickle_path) == False:
		print('■ step 4. creating operator pickle file .... ')
		dp.create_operator_vocab_pickle(self.operator_dict)
	else:
		print('■ step 4. operator pickle file exists .... ')
		#operator_dict = dp.load_pickle(hp.operator_pickle_path)

if __name__ == '__main__':
	print("■ To create vocabulary & pickle files .... ")
	tf.app.run()