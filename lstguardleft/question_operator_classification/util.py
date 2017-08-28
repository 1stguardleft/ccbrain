import numpy as np

def MinMaxScaler(data):
	''' Min Max Normalization

	Parameters
	----------
	data : numpy.ndarray
		input data to be normalized
		shape: [Batch size, dimension]

	Returns
	----------
	data : numpy.ndarry
		normalized data
		shape: [Batch size, dimension]

	References
	----------
	.. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

	'''
	numerator = data - np.min(data, 0)
	denominator = np.max(data, 0) - np.min(data, 0)
	# noise term prevents the zero division
	return numerator / (denominator + 1e-7)

def get_batches(x, y, batch_size=100):
	n_batches = len(x)//batch_size
	x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
	for ii in range(0, len(x), batch_size):
		yield x[ii:ii+batch_size], y[ii:ii+batch_size]

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
  """
  string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)

  return string.strip().lower()