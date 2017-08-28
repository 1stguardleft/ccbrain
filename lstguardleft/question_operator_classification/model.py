# model.py
# 해결하기 위한 문제에 대한 Computation Graph를 정의
# graph 및 TensorFlow op을 생성(build)하는 역할
# 내용
# - model 의 constructor에 session을 넣어주지 않음에 유의
# 주의
# - Checkpoint Laod & Save >> train.py의 Responsibilty
# - Training >> train.py의 Responsibilty
# - model과 session의 완벽한 decoupling이 어려운 경우 model 코드내에서는 session을 injection 하는 방식을 취함
# - input to output 매핑이 가능해야 함
# - 중간 Tensor (layer)의 activation 에 접근 가능해야 함
# - Model에 존재하는 variable을 가지고 올 수 있어야 함
# - 작성한 Component를 library 처럼 다른 곳에 붙이거나 여러 모델을 합성하기 용이해야 함

import tensorflow as tf
import config as cf

class QOClassifier:
	# 클래스 변수 = model의 각종 OP들
	# 실제 Model을 참조하는 train, eval 에서 필요한 OP 들만 우선 선정
	# 클래스 변수 = 환경변수 NamedTuple
	def __init__(self, config):  # config는 각종 hyper-parameter를 정의하는 객체
		self.config = config

		self.cell = None
		self.inital_state = None
		self.cost = None
		self.input = None
		self.label = None
		self.keep_prob = None
		self.fc_output = None
		self.final_state = None
		self.label_one_hot = None

		self.define_model()

	def define_model(self):
		# Create the graph object
		tf.reset_default_graph()

		# Logging 설정 - 
		tf.logging.set_verbosity(tf.logging.INFO) # TOOD 1 : logging option 또한 config로 가지고 갈 것

		with tf.name_scope('Input'):
			self.input = tf.placeholder(tf.int32, [None, None], name="input")
			self.label = tf.placeholder(tf.int32, [None], name="label")
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

		with tf.name_scope("Embeddings"):
			embedding_input = tf.Variable(tf.random_uniform((self.config.vocab_size, self.config.embedding_dim), -1, 1))
			embed_input = tf.nn.embedding_lookup(embedding_input, self.input)

			self.label_one_hot = tf.one_hot(self.label, self.config.label_classes, 1, 0)

		with tf.name_scope("RNN_layers"):
			# Stack up multiple LSTM layers, for deep learning
			self.cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.config.lstm_layer)])

			# Getting an initial state of all zeros
			self.initial_state = self.cell.zero_state(self.config.batch_size, tf.float32)

		with tf.name_scope("RNN_forward"):
			rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, embed_input, initial_state=self.initial_state)

		with tf.name_scope('predictions'):
			self.fc_output = tf.contrib.layers.fully_connected(rnn_outputs[:, -1], self.config.label_classes, activation_fn=None)

			# 횡단관심 (Cross-cutting Concerns) 
			# 각 레이어(파일) 마다 발생하는 summary, loss, variable등을 collection으로 관리
			# Collection은 tf.graph에 전역으로 관리되는 Singleton임
			tf.summary.histogram('logits', self.fc_output)
		
		with tf.name_scope('cost'):
			cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_output, labels=tf.to_float(self.label_one_hot))
			self.cost = tf.reduce_sum(cost)

			tf.summary.scalar('cost', self.cost)

	def lstm_cell(self):
		#lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
		lstm = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_size)
		# Add dropout to the cell
		return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.config.train_keep_prob)

	def predict(self):
		pass
	#def inference(images, num_classes, for_training=False, restore_logits=True, scope=None):