from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd

import preprocessor as pr
import util as ut
from tensorflow.contrib import rnn

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from konlpy.tag import Kkma
from konlpy.tag import Twitter

# TODO 6.1 : TODO 1 ~ 5 까지 데이터 Load (using by Pickle)

vocab_dict = pr.load_pickle('./data/question_dict.pickle')
operator_dict = pr.load_pickle('./data/operator_dict.pickle')

# TODO 6.2 : Raw Data를 읽어들여 Dictionary와 Mapping하여 관리 (Datatype : nested list)

epochs        = 1   # 1 epoch = one forward pass and one backward pass of all the  training examples.
learing_rate  = 0.01
batch_size    = 1  # batch size = the number of training examples in one forward/backward pass.
iterations    = 2  # tot_data_size / batch_size = iterations = number of passes, each pass using [batch_size] number of examples.
tot_data_size = batch_size * iterations

input_dim   = len(vocab_dict)
num_classes = len(vocab_dict)

# Hidden Size = num_units = output sequence의 길이(?)
hidden_size = 1  # Many to one model
hidden_dim  = 10 #
'''
[example]
if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.
'''

dataX = []
dataY = []
twitter = Twitter()

df_sop = pd.read_csv('./data/question_operator_set_1500.csv')

for idx, value in enumerate(df_sop['question']):
    dataX.append([vocab_dict[question] for question in twitter.morphs(value)]) 
    
for idx, value in enumerate(df_sop['operator']):
    dataY.append([operator_dict[value]])   

dataY = ut.MinMaxScaler(dataY)

# TODO 6.3 : DynamicRNN 예제에 맞게 데이터 변형 이후 수행

# sequence_length = RNN 모델의 cell의 갯수 = 한 개 Question의 길이
# sequence_lenght가 가변적인 경우 각 배치의 길이를 리스트로 지정
sequence_length_list = [len(val) for idx, val in enumerate(dataX)]
sequence_length = max_len = max(sequence_length_list)

# Zero-padding, 단 Zero-Padding시에는 nn.dynamic_rnn 사용을 하지 않음
for idx, val in enumerate(dataX):
    if len(dataX[idx]) < max_len:
        for ind in range(0, max_len - len(dataX[idx])):
            dataX[idx].append(0)

#dataY의 경우의 수를 구한다.  
operator_num = 99

# dataX를 one-hot-encoding 수행
ohe = OneHotEncoder()
ohe.dtype = np.float32

arrX = np.array(dataX)
arrX1d = arrX.reshape(-1, 1)

ohe.fit(arrX1d)
ohe.n_values_ = input_dim

one_hot_arr = ohe.transform(arrX1d).toarray()
X_one_hot = one_hot_arr.reshape(-1, 50, 2047) 

# dataY를 one-hot-encoding수행
'''
arrY = np.array(dataY)

ohe.fit(arrY)
ohe.n_values_ = operator_num
Y_one_hot = ohe.transform(arrY).toarray()
'''

# Hyper parameter 출력
print('========================================')
print('input data dimension : '                    + str(input_dim)  )
print('length of output sequence (hidden_size) : ' + str(hidden_size))
print('num_classes : '                             + str(num_classes))
print('batch_size : '                              + str(batch_size) )
print('sequence_length : '                              + str(sequence_length) )
print('========================================')

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.float32, [None, hidden_size])               # Many to one Model

# One-hot encoding
# X_one_hot = tf.one_hot(dataX, num_classes)

# check out the shape
# print(X_one_hot.shape)  

# Make a lstm cell with hidden_size (each unit output vector size)
cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple = True)
cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple = True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# fully_connected layer
#Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], hidden_dim, activation_fn=None)

# cost/loss
loss = tf.reduce_sum(tf.square(outputs[:, -1] - Y))  # sum of the squares

# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations): # iterations = 15 = 1500 / 100
        _, step_loss = sess.run([train, loss], feed_dict={X: X_one_hot[i*batch_size:(i+1)*batch_size], Y: dataY[i*batch_size:(i+1)*batch_size]})
        result = sess.run(outputs, feed_dict={X: X_one_hot[i*batch_size:(i+1)*batch_size]})
        print("[step: {} loss: {} output : {} Y_real : {}".format(i, step_loss, result, dataY[i]))
