"""Train stuff."""
import collections
import os
import numpy as np
import tensorflow as tf

data_type = tf.float32

use_sms = True
do_tv = False

if not use_sms:
  DATA_DIR = 'data'
  VOCAB = TRAIN = 'train.txt'
else:
  DATA_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/out'
  VOCAB = TRAIN = 'sms-20171118000041.xml_se.train.txt'
  TEST = 'sms-20171118000041.xml_se.test.txt'
  VALID = 'sms-20171118000041.xml_se.valid.txt'
INIT_SCALE = 0.1

SAVE_PATH = '/Users/jiayu/Documents/1Stanford/cs229/project/taffy/seq/out/all'


class DefaultConfig(object):
  num_hidden_units = 50
  num_layers = 2
  dropout = 0.2
  num_steps = 10
  learning_rate = 1.00
  batch_size = 10


class SeqModel(object):
  def __init__(self, cfg, data, is_training=False):
    self.num_hidden_units = cfg.num_hidden_units
    self.num_layers = cfg.num_layers
    self.dropout = cfg.dropout
    self.learning_rate = cfg.learning_rate
    self.data = data
    self.eval_op = None  # Will be none for non-training models.

    self._init_rnn(is_training)

  def _init_rnn(self, is_training):
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

    # Describe the RNN training model.
    self.cost = cost
    self.initial_state = initial_state
    self.final_state = final_state
    if not is_training:
      return
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=self.learning_rate)
    self.eval_op = optimizer.minimize(cost)


def run(model, sess):
  state = sess.run(model.initial_state)

  training_graphs = {
      'state': model.final_state,
      'cost': model.cost,
  }
  if model.eval_op is not None:
    training_graphs['eval_op'] = model.eval_op

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
      print('Step: %s, cost: %.03f, ppl: %0.3f' % (
          step, cost, np.exp(costs / iters)))

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

    with tf.name_scope(filename, 'Reader', [
        self.data_ids[filename], self.batch_size, self.num_steps]):
      return self._set_file_data(filename, self.batch_size, self.num_steps)

  def _set_file_data(self, filename, batch_size, num_steps):
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

    return Data(
        x=x, y=y, batch_size=batch_size, num_steps=num_steps,
        vocab_size=self.vocab_size, num_data=len(ids))


def main(_):
  cfg = DefaultConfig()

  with tf.Graph().as_default():

    reader = DataReader(DATA_DIR, VOCAB, cfg)

    initializer = tf.random_uniform_initializer(-INIT_SCALE, INIT_SCALE)

    with tf.name_scope('Train'):
      with tf.variable_scope('MyRNN', initializer=initializer, reuse=None):
        train_data = reader.set_file_data(TRAIN)
        seq_model = SeqModel(cfg, train_data, is_training=True)
      tf.summary.scalar('Training loss', seq_model.cost)

    if do_tv:
      with tf.name_scope('Validate'):
        with tf.variable_scope('MyRNN', initializer=initializer, reuse=True):
          valid_data = reader.set_file_data(VALID)
          valid_seq_model = SeqModel(cfg, valid_data)
        tf.summary.scalar('Validation loss', valid_seq_model.cost)

      with tf.name_scope('Test'):
        test_cfg = DefaultConfig()
        test_cfg.batch_size = 1
        test_cfg.num_steps = 1
        with tf.variable_scope('MyRNN', initializer=initializer, reuse=True):
          test_data = reader.set_file_data(TEST)
          test_seq_model = SeqModel(cfg, test_data)
        tf.summary.scalar('Test loss', valid_seq_model.cost)#'''

    a = '''
    metagraph = tf.train.export_meta_graph()

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)'''

    sv = tf.train.Supervisor(logdir='/tmp/tf/log1')
    config_proto = tf.ConfigProto(allow_soft_placement=False)
    with sv.managed_session(config=config_proto) as sess:
      for i in range(2):
        perplexity = run(seq_model, sess)
        print('%s: Training perplexity: %.3f' % (i, perplexity))
        if do_tv:
          valid_perplexity = run(valid_seq_model, sess)
          print('%s: Validation perplexity: %.3f' % (i, valid_perplexity))

      if do_tv:
        test_perplexity = run(test_seq_model, sess)
        print('Test perplexity: %.3f' % test_perplexity)

      if SAVE_PATH:
        print('Saving model to: %s' % SAVE_PATH)
        sv.saver.save(sess, SAVE_PATH, global_step=sv.global_step)


if __name__ == '__main__':
  tf.app.run()
