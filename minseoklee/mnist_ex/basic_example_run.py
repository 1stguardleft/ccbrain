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

input_data = [[0.05, 0.1], [0.05, 0.1]]
output_data = [[0.01, 0.99], [0.01, 0.99]]
input_size = len(input_data[0])
output_size = len(output_data[0])
learning_rate = 0.5

hidden_layer_neuron_size = [2]
  
x = tf.placeholder("float", [None, input_size])
y_ = tf.placeholder("float", [None, output_size])

#W, b = ffn.fead_forward_nn_init(input_size, output_size, hidden_layer_neuron_size)

W = [tf.Variable([[0.15, 0.25], [0.20, 0.30]]), tf.Variable([[0.4, 0.5], [0.45, 0.55]])]
b = [tf.Variable([0.35, 0.35]), tf.Variable([0.6, 0.6])]

pred = ffn.fead_forward_nn(x, W, b)
cost = tf.reduce_mean(tf.pow(tf.subtract(pred, y_), 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000):
    batch_x, batch_y = np.array([input_data[0]]), np.array([output_data[0]])
    sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y})
    if i % 100 == 0:
      correct_prediction = tf.reduce_mean(tf.pow(tf.subtract(pred, output_data[0]), 2))
      print("{}th cost : ".format(i), end = "")
      print(sess.run(correct_prediction, feed_dict = {x: np.array([input_data[0]]), y_: np.array([output_data[0]])}))
sess.close()