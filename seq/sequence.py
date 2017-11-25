"""Train stuff."""
import collections
import os
import numpy as np
import tensorflow as tf

data_type = tf.float32

DATA_DIR = 'data'
VOCAB = 'train.txt'
TRAIN = 'train.txt'
INIT_SCALE = 1.0


class DefaultConfig(object):
  num_hidden_units = 100
  num_layers = 2
  dropout = 0.2
  num_steps = 20
  learning_rate = 1.00
  batch_size = 20


class SeqModel(object):
  def __init__(self, cfg, data):
    self.num_hidden_units = cfg.num_hidden_units
    self.num_layers = cfg.num_layers
    self.dropout = cfg.dropout
    self.learning_rate = cfg.learning_rate
    self.data = data

    self._init_rnn()

  def _init_rnn(self):
    cell = tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(self.num_hidden_units),
        output_keep_prob=1.0 - self.dropout)

    cell = tf.contrib.rnn.MultiRNNCell(
        [cell] * self.num_layers, state_is_tuple=True)

    # State is tuple of LSTMStateTuple's.
    initial_state = cell.zero_state(self.data.batch_size, data_type)

    # Make x embeddings.
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [self.data.vocab_size, self.num_hidden_units],
          dtype=data_type)
      inputs = tf.nn.embedding_lookup(embedding, self.data.x)
    inputs = tf.nn.dropout(inputs, 1 - self.dropout)  # If training.
    inputs = tf.unstack(inputs, num=self.data.num_steps, axis=1)

    outputs, final_state = tf.contrib.rnn.static_rnn(
        cell, inputs, initial_state=initial_state, dtype=data_type)
    output = tf.reshape(tf.concat(outputs, 1), [-1, self.num_hidden_units])
    print('output shape:', output.shape)

    # Have inputs, output, initial_state, final_state

    # Get prediction.
    w = tf.get_variable(
        "weights", [self.num_hidden_units, self.data.vocab_size],
        dtype=data_type)
    b = tf.get_variable("biases", [self.data.vocab_size], dtype=data_type)
    logits = tf.nn.xw_plus_b(output, w, b)
    logits = tf.reshape(logits, [
        self.data.batch_size, self.data.num_steps, self.data.vocab_size])

    # Measures cost between predicted logits and target y.
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        self.data.y,
        tf.ones([self.data.batch_size, self.data.num_steps], dtype=data_type),
        average_across_timesteps=False,
        average_across_batch=True)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=self.learning_rate)
    # Describes the RNN training model.
    self.cost = cost
    self.eval_op = optimizer.minimize(cost)
    self.initial_state = initial_state
    self.final_state = final_state


def run(model, sess):
  state = sess.run(model.initial_state)

  training_graphs = {
      'state': model.final_state,
      'eval_op': model.eval_op,
      'cost': model.cost,
  }

  costs = 0.0
  iters = 0

  print('Epoch size %s, data size: %s, batch size: %s' % (
      model.data.epoch_size, model.data.num_data, model.data.batch_size))
  print('Vocab size %s' % (model.data.vocab_size))
  for step in range(model.data.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c  # c = hidden state
      feed_dict[h] = state[i].h  # h = output

    vals = sess.run(training_graphs, feed_dict=feed_dict)
    cost = vals['cost']  # Update cost.
    state = vals['state']  # Update state.

    costs += cost
    iters += model.data.num_steps
    if step % 10 == 0:
      print('Step: %s, ppl: %0.3f' % (step, np.exp(costs / iters)))

  return np.exp(costs / iters)  # Perplexity.


class Data(object):
  def __init__(self, x, y, batch_size, num_steps, vocab_size, num_data):
    self.x = x
    self.y = y
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.vocab_size = vocab_size
    self.num_data = num_data
    self.epoch_size = ((num_data // batch_size) - 1) // num_steps


class DataReader(object):

  EOS_TOKENS = ['\n', '.']
  EOS = '<eos>'

  def __init__(self, dir_path, vocab_file, cfg):
    self.dir_path = dir_path
    self._make_vocabulary(vocab_file)
    self.data_ids = {}
    self.batch_size = cfg.batch_size
    self.num_steps = cfg.num_steps
    self.datas = {}

  def _read_words(self, filename):
    filename = os.path.join(self.dir_path, filename)
    print('Reading file:', filename)
    with tf.gfile.GFile(filename, "r") as f:
      contents = f.read()
      for token in self.EOS_TOKENS:
        contents = contents.replace(token, self.EOS)
      return contents.split()

  def _make_vocabulary(self, filename):
    vocab_words = self._read_words(filename)
    counter = collections.Counter(vocab_words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    self.vocab = dict(zip(words, range(len(words))))
    self.vocab_size = len(self.vocab)

  def set_file_ids(self, filename):
    words = self._read_words(filename)
    ids = [self.vocab[word] for word in words if word in self.vocab]
    self.data_ids[filename] = ids

  def set_file_data(self, filename):
    if filename not in self.datas:
      self.set_file_ids(filename)

    batch_size = self.batch_size
    num_steps = self.num_steps

    ids = self.data_ids[filename]
    data = tf.convert_to_tensor(ids, name=filename, dtype=tf.int32)

    data_len = tf.size(data)
    batch_len = data_len // batch_size
    data = tf.reshape(data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    # Y is X time-shifted by 1.
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])

    self.datas[filename] = Data(
        x=x, y=y, batch_size=batch_size, num_steps=num_steps,
        vocab_size=self.vocab_size, num_data=len(ids))


def main(_):
  cfg = DefaultConfig()

  with tf.Graph().as_default():

    reader = DataReader(DATA_DIR, VOCAB, cfg)
    reader.set_file_data(TRAIN)

    initializer = tf.random_uniform_initializer(-INIT_SCALE, INIT_SCALE)

    with tf.variable_scope('MyRNN', initializer=initializer, reuse=None):
      seq_model = SeqModel(cfg, reader.datas[TRAIN])

    a = '''
    metagraph = tf.train.export_meta_graph()

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)'''

    sv = tf.train.Supervisor(logdir='/tmp/tf/log1')
    config_proto = tf.ConfigProto(allow_soft_placement=False)
    with sv.managed_session(config=config_proto) as sess:
      for _ in range(2):
        perplexity = run(seq_model, sess)
        print('Momo perplexity: %.3f' % perplexity)


if __name__ == '__main__':
  tf.app.run()
