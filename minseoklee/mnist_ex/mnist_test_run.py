'''
이 예제는 해당 사이트에 있는 전형적인 Neural Network를 구현한 예제이다.
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
실행 결과는 예제와 모두 동일하게 나타났다.
input의 크기는 이며, output의 크기도 2이다.
Hidden Layer의 크기는 1이며, 첫번째 Hidden Layer의 Neuron 크기는 2이다.
Non-linear Function으로는 tanh 사용하였다.
'''

import tensorflow as tf
import fead_forward_nn as ffn
import numpy as np
import json as json
import pickle as pickle
import os as os
import shutil

from tensorflow.examples.tutorials.mnist import input_data

def file_save(w, b, iter):
  out_W = open('./mnist_result/data_w.bak', 'wb')
  out_b = open('./mnist_result/data_b.bak', 'wb')
  out_iter = open('./mnist_result/data_iter.bak', 'wb')
  pickle.dump(w, out_W)
  pickle.dump(b, out_b)
  pickle.dump(iter, out_iter)
  out_W.close()
  out_b.close()
  out_iter.close()
  shutil.copy2('./mnist_result/data_w.bak', './mnist_result/data_w')
  shutil.copy2('./mnist_result/data_b.bak', './mnist_result/data_b')
  shutil.copy2('./mnist_result/data_iter.bak', './mnist_result/data_iter')
  os.remove('./mnist_result/data_w.bak')
  os.remove('./mnist_result/data_b.bak')
  os.remove('./mnist_result/data_iter.bak')


mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

input_data = mnist.test.images
output_data = mnist.test.labels
input_size = len(input_data[0])
output_size = len(output_data[0])
learning_rate = 0.001
display_step = 5

training_epochs = 30
batch_size = 100

hidden_layer_neuron_size = [256, 256]

x = tf.placeholder("float", [None, input_size])
y = tf.placeholder("float", [None, output_size])

W = []
b = []
iter = 0

if os.path.exists("./mnist_result/data_w") == True and os.path.exists("./mnist_result/data_b") == True and os.path.exists("./mnist_result/data_iter") == True :
  out_W = open('./mnist_result/data_w', 'rb')
  W_ = pickle.load(out_W)
  for i in range(len(W_)):
    W = W + [tf.Variable(W_[i])]
  out_b = open('./mnist_result/data_b', 'rb')
  b_ = pickle.load(out_b)
  for i in range(len(b_)):
    b = b + [tf.Variable(b_[i])]
  out_iter = open('./mnist_result/data_iter', 'rb')
  iter = pickle.load(out_iter)
else :
#  W, b = ffn.fead_forward_nn_init_with_zeros(input_size, output_size, hidden_layer_neuron_size)
#  W, b = ffn.fead_forward_nn_init_with_zero_normal(input_size, output_size, hidden_layer_neuron_size)
  W, b = ffn.xavier_init(input_size, output_size, hidden_layer_neuron_size)

y_ = ffn.fead_forward_nn(x, W, b)
cost = tf.reduce_mean(tf.pow(tf.subtract(y_, y), 2))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_, labels = y))
#cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices = 1))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch) :
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      _, W_result, b_result, cost_result = sess.run([optimizer, W, b, cost], feed_dict = {x: batch_x, y: batch_y})
      avg_cost += cost_result / total_batch
      iter = iter + batch_size
    if epoch % display_step == 0:
      y_result_set = ffn.fead_forward_nn(input_data, W, b)
      correct_prediction = tf.equal(tf.argmax(y_result_set, 1), tf.argmax(output_data, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      log_file = open('./log_files', 'a')
      print("{}th cost : {:.9f}".format(iter + 1, avg_cost))
      log_file.write("{}th cost : {}\n".format(iter + 1, avg_cost))
      print("{}th accuracy : {:.4f}%".format(iter + 1, accuracy.eval() * 100.))
      log_file.write("{}th accuracy : {:.4f}%\n".format(iter + 1, accuracy.eval() * 100.))
      log_file.close()
      file_save(W_result, b_result, iter)
      print("{}th file write success!!".format(iter + 1))
  print("Optimization Finished!")
  y_result_set = ffn.fead_forward_nn(input_data, W, b)
  correct_prediction = tf.equal(tf.argmax(y_result_set, 1), tf.argmax(output_data, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  log_file = open('./log_files', 'a')
  print("{}th cost : {:.9f}".format(iter + 1, avg_cost))
  log_file.write("{}th cost : {}\n".format(iter + 1, avg_cost))
  print("{}th accuracy : {:.4f}%".format(iter + 1, accuracy.eval() * 100.))
  log_file.write("{}th accuracy : {:.4f}%\n".format(iter + 1, accuracy.eval() * 100.))
  log_file.close()
  file_save(W_result, b_result, iter)
  print("{}th file write success!!".format(iter + 1))
