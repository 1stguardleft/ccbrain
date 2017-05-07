import tensorflow as tf
import math as math

def fead_forward_nn_init_with_zeros(input_size, output_size, hidden_layer_neuron_size):
  W = []
  b = []

  if len(hidden_layer_neuron_size) != 0 :
    W = W + [tf.Variable(tf.zeros([input_size, hidden_layer_neuron_size[0]]))]
    b = b + [tf.Variable(tf.zeros([hidden_layer_neuron_size[0]]))]
  else :
    W = W + [tf.Variable(tf.zeros([input_size, output_size]))]
    b = b + [tf.Variable(tf.zeros([output_size]))]

  for i in range(1, len(hidden_layer_neuron_size)) :
    W = W + [tf.Variable(tf.zeros([hidden_layer_neuron_size[i - 1], hidden_layer_neuron_size[i]]))]
    b = b + [tf.Variable(tf.zeros([hidden_layer_neuron_size[i]]))]

  if len(hidden_layer_neuron_size) != 0 :
    W = W + [tf.Variable(tf.zeros([hidden_layer_neuron_size[len(hidden_layer_neuron_size) - 1], output_size]))]
    b = b + [tf.Variable(tf.zeros([output_size]))]

  return W, b

def fead_forward_nn_init_with_zero_normal(input_size, output_size, hidden_layer_neuron_size):
  W = []
  b = []

  if len(hidden_layer_neuron_size) != 0 :
    W = W + [tf.Variable(tf.random_normal([input_size, hidden_layer_neuron_size[0]]))]
    b = b + [tf.Variable(tf.random_normal([hidden_layer_neuron_size[0]]))]
  else :
    W = W + [tf.Variable(tf.random_normal([input_size, output_size]))]
    b = b + [tf.Variable(tf.random_normal([output_size]))]

  for i in range(1, len(hidden_layer_neuron_size)) :
    W = W + [tf.Variable(tf.random_normal([hidden_layer_neuron_size[i - 1], hidden_layer_neuron_size[i]]))]
    b = b + [tf.Variable(tf.random_normal([hidden_layer_neuron_size[i]]))]

  if len(hidden_layer_neuron_size) != 0 :
    W = W + [tf.Variable(tf.random_normal([hidden_layer_neuron_size[len(hidden_layer_neuron_size) - 1], output_size]))]
    b = b + [tf.Variable(tf.random_normal([output_size]))]

  return W, b

def fead_forward_nn_init_with_normal(input_size, output_size, hidden_layer_neuron_size, expect, variance):
  W = []
  b = []

  if len(hidden_layer_neuron_size) != 0 :
    W = W + [tf.Variable(tf.divide(tf.add(tf.random_normal([input_size, hidden_layer_neuron_size[0]]), expect), variance))]
    b = b + [tf.Variable(tf.divide(tf.add(tf.random_normal([hidden_layer_neuron_size[0]]), expect), variance))]
  else :
    W = W + [tf.Variable(tf.divide(tf.add(tf.random_normal([input_size, output_size]), expect), variance))]
    b = b + [tf.Variable(tf.divide(tf.add(tf.random_normal([output_size]), expect), variance))]

  for i in range(1, len(hidden_layer_neuron_size)) :
    W = W + [tf.Variable(tf.divide(tf.add(tf.random_normal([hidden_layer_neuron_size[i - 1], hidden_layer_neuron_size[i]]), expect), variance))]
    b = b + [tf.Variable(tf.divide(tf.add(tf.random_normal([hidden_layer_neuron_size[i]]), expect), variance))]

  if len(hidden_layer_neuron_size) != 0 :
    W = W + [tf.Variable(tf.divide(tf.add(tf.random_normal([hidden_layer_neuron_size[len(hidden_layer_neuron_size) - 1], output_size]), expect), variance))]
    b = b + [tf.Variable(tf.divide(tf.add(tf.random_normal([output_size]), expect), variance))]

  return W, b

def fead_forward_nn(x, W, b) :
  x_ = x
  for i in range(len(W) - 1) :
    #x_ = tf.nn.tanh(tf.add(tf.matmul(x_, W[i]), b[i]))
    x_ = tf.nn.relu(tf.add(tf.matmul(x_, W[i]), b[i]))
  x_ = tf.add(tf.matmul(x_, W[len(W) - 1]), b[len(W) - 1])
  return x_

def xavier_init(input_size, output_size, hidden_layer_neuron_size, uniform = True) :
  W = []
  b = []

  if (uniform == False) :
    return fead_forward_nn_init_with_normal(input_size, output_size, hidden_layer_neuron_size, 0., tf.sqrt(3.0 / (input_size + output_size)))
  if len(hidden_layer_neuron_size) != 0 :
    init_range = math.sqrt(6.0 / (input_size + hidden_layer_neuron_size[0]))
    W = W + [tf.Variable(tf.random_uniform([input_size, hidden_layer_neuron_size[0]], minval = -init_range, maxval = init_range))]
    b = b + [tf.Variable(tf.zeros([hidden_layer_neuron_size[0]]))]
  else :
    init_range = math.sqrt(6.0 / (input_size + output_size))
    W = W + [tf.Variable(tf.random_uniform([input_size, output_size], minval = -init_range, maxval = init_range))]
    b = b + [tf.Variable(tf.zeros([output_size]))]

  for i in range(1, len(hidden_layer_neuron_size)) :
    init_range = math.sqrt(6.0 / (hidden_layer_neuron_size[i - 1] + hidden_layer_neuron_size[i]))
    W = W + [tf.Variable(tf.random_uniform([hidden_layer_neuron_size[i - 1], hidden_layer_neuron_size[i]], minval = -init_range, maxval = init_range))]
    b = b + [tf.Variable(tf.zeros([hidden_layer_neuron_size[i]]))]

  if len(hidden_layer_neuron_size) != 0 :
    init_range = math.sqrt(6.0 / (hidden_layer_neuron_size[len(hidden_layer_neuron_size) - 1] + output_size))
    W = W + [tf.Variable(tf.random_uniform([hidden_layer_neuron_size[len(hidden_layer_neuron_size) - 1], output_size], minval = -init_range, maxval = init_range))]
    b = b + [tf.Variable(tf.zeros([output_size]))]

  return W, b
    