'''
이 예제는 해당 사이트에 있는 전형적인 Neural Network를 구현한 예제이다.
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
실행 결과는 예제와 모두 동일하게 나타났다.
input의 크기는 이며, output의 크기도 2이다.
Hidden Layer의 크기는 1이며, 첫번째 Hidden Layer의 Neuron 크기는 2이다.
Non-linear Function으로는 Sigmoid를 사용하였다.
'''

import tensorflow as tf
import fead_forward_nn as ffn
import numpy as np
import json as json
import pickle as pickle
import os as os
import shutil

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

input_data = mnist.test.images
output_data = mnist.test.labels
input_size = len(input_data[0])
output_size = len(output_data[0])
learning_rate = 0.001

hidden_layer_neuron_size = [784, 784, 784, 784]

x = tf.placeholder("float", [None, input_size])
y_ = tf.placeholder("float", [None, output_size])

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
  W, b = ffn.fead_forward_nn_init_with_zero_normal(input_size, output_size, hidden_layer_neuron_size)

pred = ffn.fead_forward_nn(x, W, b)
cost = tf.reduce_mean(tf.pow(tf.subtract(pred, y_), 2))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000000):
    batch_x, batch_y = mnist.train.next_batch(1)
    sess.run(optimizer, feed_dict = {x: batch_x, y_: batch_y})
    iter = iter + 1
    if iter % 1000 == 0:
      pred_ = ffn.fead_forward_nn(x, W, b)
      correct_prediction = tf.reduce_mean(tf.pow(tf.subtract(pred_, y_), 2))
      print("{}th cost : ".format(iter + 1), end = "")
      cost2 = sess.run(correct_prediction, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
      print(cost2)
      log_file = open('./log_files', 'a')
      log_file.write("{}th cost : {}\n".format(iter + 1, cost2))
      y_result = ffn.fead_forward_nn(x, W, b)
      correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_result, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      print("{}th accuracy : ".format(iter + 1), end = "")
      accuracy2 = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
      print("{}%".format(accuracy2 * 100))
      log_file.write("{}th accuracy : {}%\n".format(iter + 1, accuracy2 * 100))
      log_file.close()
      out_W = open('./mnist_result/data_w.bak', 'wb')
      out_b = open('./mnist_result/data_b.bak', 'wb')
      out_iter = open('./mnist_result/data_iter.bak', 'wb')
      accuracy_result, W_result, b_result = sess.run([accuracy, W, b], feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
      pickle.dump(W_result, out_W)
      pickle.dump(b_result, out_b)
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
      print("{}th file write success!!".format(iter + 1))
  y_result = ffn.fead_forward_nn(x, W, b)
  correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_result, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  out_W = open('./mnist_result/data_w.bak', 'wb')
  out_b = open('./mnist_result/data_b.bak', 'wb')
  out_iter = open('./mnist_result/data_iter.bak', 'wb')
  accuracy_result, W_result, b_result = sess.run([accuracy, W, b], feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
  pickle.dump(W_result, out_W)
  pickle.dump(b_result, out_b)
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
  print("final accuracy : {}".format(accuracy_result))
  sess.close()