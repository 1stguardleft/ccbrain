import tensorflow as tf
from collections import namedtuple

# Data Processing Parameters
tf.flags.DEFINE_string("train_file_path", "./data/question_operator_set_1500.csv", "Path to train file")
tf.flags.DEFINE_string("question_vocab_path", "./data/question_vocab.txt", "Path to question vocabulary file")
tf.flags.DEFINE_string("question_pickle_path", "./data/question_dict.pickle", "Path to question pickle file")
tf.flags.DEFINE_string("operator_vocab_path", "./data/operator_vocab.txt", "ath to operator vocabulary")
tf.flags.DEFINE_string("operator_pickle_path", "./data/operator_dict.pickle", "Path to operator pickle file")

tf.flags.DEFINE_integer("vocab_size", 2048, "The size of the vocabulary. Only change this if you changed the preprocessing")
tf.flags.DEFINE_integer("label_classes", 99, "The size if the number of labels")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("lstm_layer", 1, "the number of lstm layer")
tf.flags.DEFINE_integer("lstm_size", 512, "Dimensionality of the RNN cell")           # hidden_dim과 동일
tf.flags.DEFINE_integer("max_context_len", 50, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 80, "Truncate utterance to this length") # 현재 미사용이나 이 기능 추가 필요

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", None, "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", "./data/question_vocab.txt", "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.flags.DEFINE_integer("epochs", 10, "epochs number")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 16, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
tf.flags.DEFINE_string("activation_fn", "None", "activation function of FC layser's output")
tf.flags.DEFINE_float("train_keep_prob", 1.0, "drop-out rate")

# Validation Parameters 
tf.flags.DEFINE_string("train_dir", "./checkpoints", "학습데이터 저장 디렉토리")

#tf.flags.DEFINE_float("val_keep_prob", 0.9, "")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
	"HParams",
	[
		"train_file_path",
		"question_vocab_path",
		"question_pickle_path",
		"operator_vocab_path",
		"operator_pickle_path",
		"vocab_size",
		"label_classes",
		"embedding_dim",
		"lstm_layer",
		"lstm_size",
		"max_context_len",
		"max_utterance_len",
		"glove_path",
		"vocab_path",
		"learning_rate",
		"epochs",
		"batch_size",
		"eval_batch_size",
		"optimizer",
		"activation_fn",
		"train_keep_prob",
		"train_dir"
	]
)

def create_hparams():
	return HParams(
					train_file_path=FLAGS.train_file_path,
					question_vocab_path=FLAGS.question_vocab_path,
					question_pickle_path=FLAGS.question_pickle_path,
					operator_vocab_path=FLAGS.operator_vocab_path,
					operator_pickle_path=FLAGS.operator_pickle_path,
					vocab_size=FLAGS.vocab_size,
					label_classes=FLAGS.label_classes,
					embedding_dim=FLAGS.embedding_dim,
					lstm_layer=FLAGS.lstm_layer,
					lstm_size=FLAGS.lstm_size,
					max_context_len=FLAGS.max_context_len,
					max_utterance_len=FLAGS.max_utterance_len,
					glove_path=FLAGS.glove_path,
					vocab_path=FLAGS.vocab_path,
					learning_rate=FLAGS.learning_rate,
					epochs=FLAGS.epochs,
					batch_size=FLAGS.batch_size,
					eval_batch_size=FLAGS.eval_batch_size,
					optimizer=FLAGS.optimizer,
					activation_fn=FLAGS.activation_fn,
					train_keep_prob=FLAGS.train_keep_prob,
					train_dir=FLAGS.train_dir
					)

def print_hparams(hparams):
	print("""■ vocab_size = {}\n■ label_classes = {}\n■ embedding_dim = {}\n■ embedding_dim = {}
			\n■ lstm_size = {}\n■ max_context_len = {}\n■ max_utterance_len = {}\n■ glove_path = {}
			\n■ vocab_path = {}\n■ learning_rate = {}\n■ epochs = {}\n■ batch_size = {}\n■ eval_batch_size = {}
			\n■ optimizer = {}\n■ activation_fn = {}\n■ train_keep_prob = {}\n■ train_keep_prob = {}""".format(
			hparams.vocab_size,
			hparams.label_classes,
			hparams.embedding_dim,
			hparams.lstm_layer,
			hparams.lstm_size,
			hparams.max_context_len,
			hparams.max_utterance_len,
			hparams.glove_path,
			hparams.vocab_path,
			hparams.learning_rate,
			hparams.epochs,
			hparams.batch_size,
			hparams.eval_batch_size,
			hparams.optimizer,
			hparams.activation_fn,
			hparams.train_keep_prob,
			hparams.train_dir
			)
	)