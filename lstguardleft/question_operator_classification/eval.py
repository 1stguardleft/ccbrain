import tensorflow as tf
import numpy as np

import model as md
import config as cf

from util import get_batches
from train import QOTrainer
from data import DataProcessor

class Evaluator:
	#클래스 변수
	def __init__(self, model, config):
		self.model = model
		self.config = config	
		self.accuracy = None

		self.define_accuracy()

	def define_accuracy(self):
		with tf.name_scope('evaluation'):
			correct = tf.equal(tf.argmax(self.model.fc_output,1), tf.argmax(self.model.label_one_hot, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct,'float'))

	# train 완료 이후 1번만 수행, train과 동시에 수행되는 모델도 고려해볼 필요 있음
	def evaluate_model(self, val_x, val_y):

		saver = tf.train.Saver()

		with tf.Session() as sess:
			#test_writer = tf.summary.FileWriter('./logs/eval', sess.graph)
			ckpt = tf.train.get_checkpoint_state(self.config.train_dir)
			print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)

			val_acc = []
			val_state = sess.run(self.model.cell.zero_state(self.config.batch_size, tf.float32))

			for (x, y) in get_batches(val_x, val_y, self.config.batch_size):
				feed = {self.model.input: x, self.model.label: y, self.model.keep_prob: 1.0, self.model.initial_state: val_state}
				print(self.model.label)
				print(y)
				batch_acc, val_state = sess.run([self.accuracy, self.model.final_state], feed_dict=feed)
				val_acc.append(batch_acc)

				print("Val acc: {:.3f}".format(np.mean(val_acc)))
			
def main(_):
	hp = cf.create_hparams()

	dp = DataProcessor(hp)
	model = md.QOClassifier(hp)

	evaluator = Evaluator(model, hp)
	evaluator.evaluate_model(dp.val_x, dp.val_y)

if __name__ == "__main__":
	tf.app.run()
