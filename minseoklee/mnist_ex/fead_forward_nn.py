import tensorflow as tf

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
    W = W + [tf.Variable(tf.divide(tf.add(tf.random_normal([input_size, hidden_layer_neuron_size[0]])), expect), variance)]
    b = b + [tf.Variable(tf.divide(tf.add(tf.random_normal([hidden_layer_neuron_size[0]])), expect), variance)]
  else :
    W = W + [tf.Variable(tf.divide(tf.add(tf.random_normal([input_size, output_size])), expect), variance)]
    b = b + [tf.Variable(tf.divide(tf.add(tf.random_normal([output_size])), expect), variance)]

  for i in range(1, len(hidden_layer_neuron_size)) :
    W = W + [tf.Variable(tf.divide(tf.add(tf.random_normal([hidden_layer_neuron_size[i - 1], hidden_layer_neuron_size[i]])), expect), variance)]
    b = b + [tf.Variable(tf.divide(tf.add(tf.random_normal([hidden_layer_neuron_size[i]])), expect), variance)]

  if len(hidden_layer_neuron_size) != 0 :
    W = W + [tf.Variable(tf.divide(tf.add(tf.random_normal([hidden_layer_neuron_size[len(hidden_layer_neuron_size) - 1], output_size])), expect), variance)]
    b = b + [tf.Variable(tf.divide(tf.add(tf.random_normal([output_size])), expect), variance)]

  return W, b

def fead_forward_nn(x, W, b) :
  x_ = x
  for i in range(len(W)) :
    x_ = tf.tanh(tf.add(tf.matmul(x_, W[i]), b[i]))
  return x_