import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28
X = tf.placeholder(tf.float32, [None, 784])
# 0 ~ 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters 
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples / batch_size)

		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
			avg_cost += c / total_batch

		print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

		#Test the model using test sets
		print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


