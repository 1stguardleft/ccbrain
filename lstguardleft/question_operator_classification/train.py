# train.py
# Checkpoint Laod & Save
# Training
# 내용
# - model 의 constructor에 session을 넣어주지 않음에 유의
# 주의
# - model과 session의 완벽한 decoupling이 어려운 경우 model 코드내에서는 session을 injection 하는 방식을 취함
# - input to output 매핑이 가능해야 함
# - 중간 Tensor (layer)의 activation 에 접근 가능해야 함
# - Model에 존재하는 variable을 가지고 올 수 있어야 함
# - 작성한 Component를 library 처럼 다른 곳에 붙이거나 여러 모델을 합성하기 용이해야 함

import tensorflow as tf

from data import DataProcessor
from model import QOClassifier
from config import create_hparams
from util import get_batches

class QOTrainer():
	# 클래스 변수
	def __init__(self, model, config):
		print("QOTrainer start !")

		self.config = config
		self.model = model

		self.optimizer = None
		self.merged = None

		self.define_optimizer()

	def define_optimizer(self):
		with tf.name_scope('train'):
			self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.model.cost)
			self.merged = tf.summary.merge_all()
		#opt = tf.train.GradientDescentOptimizer(learning_rate)
		# gradient variable list = [ (gradient,variable) ]
		#gv = opt.compute_gradients(cost)

		# transformed gradient variable list = [ (T(gradient),variable) ]
		#decay = 0.9 # decay the gradient for the sake of the example
		#tgv = [ (g, v) for (g,v) in gv] #list [(grad,var)]
		#tgv = [ (transform_grad(g,decay=decay), v) for (g,v) in gv] #list [(grad,var)]

		#apply transformed gradients (this case no transform)
		#apply_transform_op = opt.apply_gradients(tgv)

	def train_model(self, train_x, train_y):
		try:
			batch_size = self.config.batch_size

			# with graph.as_default():
			saver = tf.train.Saver()

			sess = tf.Session()
			sess.run(tf.global_variables_initializer())
			train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

			# Getting an initial state of all zeros
			initial_state = self.model.cell.zero_state(batch_size, tf.float32)
			iteration = 1

			for e in range(self.config.epochs):
				state = sess.run(initial_state)
				for idx, (x, y) in enumerate(get_batches(train_x, train_y, batch_size)):
					feed = {self.model.input: x, self.model.label: y, self.model.keep_prob: 0.8, initial_state: state}
					summary, losses, _ = sess.run([self.merged, self.model.cost, self.optimizer], feed_dict=feed)
			
					train_writer.add_summary(summary, iteration)

					print("Epoch: {}/{}".format(e, self.config.epochs), "Iteration: {}".format(iteration), "Train loss: {:.3f}".format(losses))
			
					iteration +=1
					saver.save(sess, "checkpoints/qo_model.ckpt")
				saver.save(sess, "checkpoints/qo_model.ckpt")
		except:
			print('do nothing ....')
		finally:
			print('training model is completed')
			sess.close()

	#funciton to transform gradients
	def transform_grad(grad, decay=1.0):
	#return decayed gradient
		return decay*grad

def main(_):
	print("main function in train.py starts")

	hp = create_hparams()

	dp = DataProcessor(hp)
	model = QOClassifier(hp)

	trainer = QOTrainer(model, hp)
	trainer.train_model(dp.train_x, dp.train_y)

if __name__ == "__main__":
	tf.app.run()