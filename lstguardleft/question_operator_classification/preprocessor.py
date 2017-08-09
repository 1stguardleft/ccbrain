import array
import os.path
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import collections

from collections import defaultdict
from pandas import Series, DataFrame

from konlpy.tag import Kkma
from konlpy.tag import Twitter

twitter = Twitter()

# TODO 1 : create_vocab 함수 정의
def create_question_vocab(train_file, vocab_file):
  vocab_list = []
  df_sop = pd.read_csv(train_file)

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

def create_operator_vocab(train_file, vocab_file):
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

def create_question_vocab_pickle(vocab_dict):
  with open('../data/question_dict.pickle', 'wb') as f:
    pickle.dump(vocab_dict, f)

def create_operator_vocab_pickle(operator_dict):
  with open('../data/operator_dict.pickle', 'wb') as f:
    pickle.dump(operator_dict, f)

def load_pickle(vocab_pickle_full_path):
  with open(vocab_pickle_full_path, 'rb') as f:
    return pickle.load(f)

def load_question_vocab(filename):
  vocab = None
  with open(filename) as f:
    vocab = f.read().splitlines()
  dct = defaultdict(int)
  for idx, word in enumerate(vocab):
    dct[word.split(':')[0]] = idx
  return dct

def load_operator_vocab(filename):
  vocab = None
  with open(filename) as f:
    vocab = f.read().splitlines()
  dct = defaultdict(int)
  for idx, word in enumerate(vocab):
    dct[word.split(':')[0]] = idx
  return dct

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
  """
  string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)

  return string.strip().lower()
  
def load_glove_vectors(filename, vocab):
  """
  Load glove vectors from a .txt file.
  Optionally limit the vocabulary to save memory. `vocab` should be a set.
  """
  dct = {}
  vectors = array.array('d')
  current_idx = 0
  with open(filename, "r", encoding="utf-8") as f:
    for _, line in enumerate(f):
      tokens = line.split(" ")
      word = tokens[0]
      entries = tokens[1:]
      if not vocab or word in vocab:
        dct[word] = current_idx
        vectors.extend(float(x) for x in entries)
        current_idx += 1
    word_dim = len(entries)
    num_vectors = len(dct)
    tf.logging.info("Found {} out of {} vectors in Glove".format(num_vectors, len(vocab)))
    return [np.array(vectors).reshape(num_vectors, word_dim), dct]


def build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, embedding_dim):
  initial_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")
  for word, glove_word_idx in glove_dict.items():
    word_idx = vocab_dict.get(word)
    initial_embeddings[word_idx, :] = glove_vectors[glove_word_idx]
  return initial_embeddings

if __name__ == "__main__":

  # TODO 2 : vocabulary 생성함수 호출하여 vocab 파일 생성
  # TODO 4.1 : Vocab 파일 존재시 Dictionary 형태로 메모리 로드 - 완료
  # TODO 4.2 : Pickle 형태로 메모리 로드 - 완료
  if os.path.isfile('../data/quetion_vocab.txt') == False:
     print('creating question vocabulary file .... ')
     vocab_dict = create_question_vocab('../data/question_operator_set_1500.csv', '../data/quetion_vocab.txt')
  else:
     vocab_dict = load_question_vocab('../data/quetion_vocab.txt')
  
  if os.path.isfile('../data/operator_vocab.txt') == False:
     print('creating operator vocabulary file .... ')
     operator_dict = create_operator_vocab('../data/question_operator_set_1500.csv', '../data/operator_vocab.txt')
  else:
     operator_dict = load_operator_vocab('../data/operator_vocab.txt')

  if os.path.isfile('../data/question_dict.pickle') == False:
     create_question_vocab_pickle(vocab_dict)
  else:
     vocab_dict = load_pickle('../data/question_dict.pickle')

  if os.path.isfile('../data/operator_dict.pickle') == False:
     create_operator_vocab_pickle(operator_dict)
  else:
     operator_dict = load_pickle('../data/operator_dict.pickle')