# 과제 1

import numpy as np
import pandas as pd

from pandas import Series, DataFrame

from konlpy.tag import Kkma
from konlpy.tag import Twitter

kkma = Kkma()
twitter = Twitter()

# csv 파일을 읽어 dataframe으로 변환
def load_sop_txt_file():
	# qo = np.loadtxt('./data/sop_sample_1500.csv', delimiter=',') 
	df_sop = pd.read_csv('./data/sop_sample_1500.csv')

	return df_sop

# dataframe에 열을 추가하고 konlpy pos tagging 하여 저장
def pos_tagging(df_sop):
	df_sop = df_sop.reindex(columns=['question', 'operator', 'pos_tag'], fill_value=0)

	for loop in range(0, df_sop.shape[0]):
	#for loop in range(0, 1):
		# lambda 인자 : 표현식
		# 축약함수 
		# (lambda x,y: x + y)(10, 20) >> 30
		# 
		df_sop['pos_tag'][loop] = ['/'.join(p) for p in twitter.pos(df_sop['question'][loop])] 
		# for tracing
		if loop == 1:
			print(df_sop['question'][loop])
			print(df_sop['pos_tag'][loop])

	return df_sop

def sava_csv_file(df_sop):
	df_sop.to_csv('./data/result.csv')

# Entry Point
if __name__ == "__main__":
	df_sop = load_sop_txt_file()
	df_sop = pos_tagging(df_sop)
	sava_csv_file(df_sop)