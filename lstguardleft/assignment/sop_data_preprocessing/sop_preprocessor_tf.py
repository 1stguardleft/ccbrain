# 과제 1
import numpy as np
import pandas as pd

from pandas import Series, DataFrame

from konlpy.tag import Kkma
from konlpy.tag import Twitter

# Using Word2Vec 
import gensim, codecs
import multiprocessing

# Word2Vec Config
config = {  
			'min_count': 5,        # ignore all words with total frequency lower than this
			'size': 300,           # dimensionality of the feature vector
			'sg': 1,               # define the training algorithm  0 : CBOW, 1 skip-gram
			'batch_words': 10,     # target size for batches of examples passed to worker threads
			'iter': 100,           # numver of iterations over the corpus
			'workers': multiprocessing.cpu_count(),
			'negative' : 15
		}

kkma = Kkma()
twitter = Twitter()

# word2vec 전환 용도
sop_vocab = []

# csv 파일을 읽어 dataframe으로 변환
def load_sop_txt_file():
	# qo = np.loadtxt('./data/sop_sample_1500.csv', delimiter=',') 
	df_sop = pd.read_csv('./data/sop_sample_1500.csv')
	#df_sop = pd.read_csv('./data/sample_1year.csv')
	
	return df_sop

# 당장 사용하지 않음.
def tokenize(doc):
	return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True) ] 

# dataframe에 열을 추가하고 konlpy pos(Part of Speech) tagging 하여 저장
def pos_tagging(df_sop):
	df_sop = df_sop.reindex(columns=['question', 'operator', 'q_pos', 'q_noun', 'q_pos_tag'], fill_value=0)
	sentences_vocab = []

	for loop in range(0, df_sop.shape[0]):
	#for loop in range(0, 1):
		# lambda 인자 : 표현식
		# 축약함수 
		# (lambda x,y: x + y)(10, 20) >> 30
		pos = twitter.pos(df_sop['question'][loop])

		df_sop['q_pos'][loop] = twitter.morphs(df_sop['question'][loop])
		df_sop['q_noun'][loop] = twitter.nouns(df_sop['question'][loop])
		df_sop['q_pos_tag'][loop] = ['/'.join(p) for p in twitter.pos(df_sop['question'][loop])]


		# for tracing
		if loop == -1:
			print(twitter.pos(df_sop['question'][loop]))
			print('question = ', df_sop['question'][loop])
			print('q_pos = ', df_sop['q_pos'][loop])
			print('q_pos_tag = ', df_sop['q_pos_tag'][loop])
			print('q_noun = ', df_sop['q_noun'][loop])

	return df_sop

# csv 파일로 저장 
def sava_csv_file(df_sop):
	df_sop.to_csv('./data/result.csv')

# Word2Vec 모델로 변환
def convert_to_word2vec(df_sop):
	sop_model = gensim.models.Word2Vec(**config)

	for i in range(0, 1500):
		sop_vocab.append(df_sop['q_pos_tag'][i])
	sop_model.build_vocab(sop_vocab)

	return sop_model

# 
# def plot_with_

# Entry Point
if __name__ == "__main__":
	df_sop = load_sop_txt_file()
	df_sop = pos_tagging(df_sop)
	sop_model = convert_to_word2vec(df_sop)

	print(sop_model.wv.most_similar('보험/Noun'))
	print(sop_model.wv.most_similar('서비스/Noun'))
	print(sop_model.wv.most_similar('명변/Noun'))