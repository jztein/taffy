# Adapted from seq2seq example.
import logging

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

import data_prep

# Toggle model things. (False, False, True, GRU) has been best.
CAP_AVG_LEN = False
PASS_ENCODER_FINAL_STATE = False
USE_ATTENTION = True  # Must be false if PASS_ENCODER_FINAL_STATE.
CELL_TYPE = 'GRU'
#CELL_TYPE = 'LSTM'

REGULARIZATION = False
OPTIMIZER = 'Adagrad'  # Adam
LEARNING_RATE = 0.1  # model_dir suffix: _001

MODEL_DIR = '/tmp/cs229/s2s/tut_movie'

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2


def seq2seq(mode, features, labels, params):
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']
    input_max_length = params['input_max_length']
    output_max_length = params['output_max_length']

    inp = features['input']
    output = features['output']
    batch_size = tf.shape(inp)[0]
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inp, 1)), 1)
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
    input_embed = layers.embed_sequence(
        inp, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    output_embed = layers.embed_sequence(
        train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')

    if CELL_TYPE == 'GRU':
        cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    else:
        cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)

    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)

    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs,
                memory_sequence_length=input_lengths)
            if CELL_TYPE == 'GRU':
                cell = tf.contrib.rnn.GRUCell(num_units=num_units)
            else:
                cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            if USE_ATTENTION:
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=num_units / 2)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, vocab_size, reuse=reuse
                )
            else:
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    cell, vocab_size, reuse=reuse
                )

            if PASS_ENCODER_FINAL_STATE and not USE_ATTENTION:
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=encoder_final_state)
            else:
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size))

            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=output_max_length
            )
            return outputs[0]

    train_op, loss, predictions = None, None, None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_outputs = decode(train_helper, 'decode')
        tf.identity(train_outputs.sample_id[0], name='train_pred')

        weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))

        loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output, output, weights=weights)

        if REGULARIZATION:
            loss += 0.01*tf.nn.l2_loss(weights)

        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer=OPTIMIZER,
            learning_rate=LEARNING_RATE,
            summaries=['loss', 'learning_rate'])
    elif mode == tf.estimator.ModeKeys.PREDICT:
        print('>>>>>>>>>>>>> yolo predict')
        infer_outputs = decode(infer_helper, 'decode', reuse=True)
        tf.identity(infer_outputs.sample_id[0], name='predictions')
        predictions = infer_outputs.sample_id 
    elif mode == tf.estimator.ModeKeys.EVALUATE:
        pass

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


def tokenize_and_map(line, vocab):
    return [vocab.get(token, UNK_TOKEN) for token in line.split(' ')]


def make_input_fn(
        batch_size, input_filename, output_filename, vocab,
        input_max_length, output_max_length,
        input_process=tokenize_and_map, output_process=tokenize_and_map):

    def input_fn():
        inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
        output = tf.placeholder(tf.int64, shape=[None, None], name='output')
        tf.identity(inp[0], 'input_0')
        tf.identity(output[0], 'output_0')
        return {
            'input': inp,
            'output': output,
        }, None

    def sampler():
        while True:
            with open(input_filename) as finput:
                with open(output_filename) as foutput:
                    for in_line in finput:
                        out_line = foutput.readline()
                        yield {
                            'input': input_process(in_line, vocab)[:input_max_length - 1] + [END_TOKEN],
                            'output': output_process(out_line, vocab)[:output_max_length - 1] + [END_TOKEN]
                        }

    feed_sample = sampler()

    def feed_fn():
        inputs, outputs = [], []
        input_length, output_length = 0, 0
        total_input_length, total_output_length = 0, 0
        for i in range(batch_size):
            rec = next(feed_sample)
            inputs.append(rec['input'])
            outputs.append(rec['output'])
            input_length = max(input_length, len(inputs[-1]))
            output_length = max(output_length, len(outputs[-1]))
            total_input_length += len(inputs[-1])
            total_output_length += len(outputs[-1])

        # Cap lengths at some number so that there are not too many noisy
        # non-word tokens.
        if CAP_AVG_LEN:
            max_input_len = max(20, int(total_input_length / batch_size) + 10)
            max_output_len = max(20, int(total_output_length / batch_size) + 10)
            max_input_len = max(max_input_len, max_output_len)
            max_output_len = max_input_len
            for i in range(batch_size):
                inputs[i] += [END_TOKEN] * (max_input_len - len(inputs[i]))
                inputs[i] = inputs[i][:max_input_len]
                outputs[i] += [END_TOKEN] * (max_output_len - len(outputs[i]))
                outputs[i] = outputs[i][:max_output_len]
        else:
            for i in range(batch_size):
                inputs[i] += [END_TOKEN] * (input_length - len(inputs[i]))
                outputs[i] += [END_TOKEN] * (output_length - len(outputs[i]))
        return {
            'input:0': inputs,
            'output:0': outputs
        }

    return input_fn, feed_fn


def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab


def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}


def get_formatter(keys, rev_vocab):

    def to_str(sequence):
        tokens = [
            rev_vocab.get(x, "<UNK>") for x in sequence]
        return ' '.join(tokens)

    def format(values):
        res = []
        for key in keys:
            res.append("%s = %s" % (key, to_str(values[key])))
        return '\n'.join(res)
    return format


def train_seq2seq(
        input_filename, output_filename, vocab_filename,
        model_dir):
    _, _, vocab, rev_vocab = data_prep.load_data_ids()
    #vocab = load_vocab(vocab_filename)
    for i, k in enumerate(vocab.keys()):
        if i > 3:
            break
        print('vlcab', k)
    params = {
        'vocab_size': len(vocab),
        'batch_size': 32,
        'input_max_length': 50,
        'output_max_length': 50,
        'embed_dim': 100,
        'num_units': 256,
    }
    print('PARAMS', params)
    est = tf.estimator.Estimator(
        model_fn=seq2seq,
        model_dir=model_dir, params=params)

    input_fn, feed_fn = make_input_fn(
        params['batch_size'],
        input_filename,
        output_filename,
        vocab, params['input_max_length'], params['output_max_length'])

    # Make hooks to print examples of inputs/predictions.
    print_inputs = tf.train.LoggingTensorHook(
        ['input_0', 'output_0'], every_n_iter=100,
        formatter=get_formatter(['input_0', 'output_0'], rev_vocab))

    print_trainoutput = tf.train.LoggingTensorHook(
        ['train_pred'], every_n_iter=100,
        formatter=get_formatter(['train_pred'], rev_vocab))

    # TODO: print this when predict().
    print_predictions = tf.train.LoggingTensorHook(
        ['predictions'], every_n_iter=100,
        formatter=get_formatter(['predictions'], rev_vocab))

    est.train(
        input_fn=input_fn,
        hooks=[tf.train.FeedFnHook(feed_fn), print_inputs, print_trainoutput],
        steps=200)

    print('Predictions:', est.predict(
            input_fn=input_fn,
            hooks=[print_predictions]))
    

OUT_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/taffy/s2s/out/'

DATA_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/taffy/s2s/out/'
X_TXT = 'movie_lines_X.txt'
Y_TXT = 'movie_lines_Y.txt'
VOCAB_TXT = 'movie_lines_vocab.txt'

#DATA_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/taffy/s2s/out/'
#X_TXT = 'sms_grouptime_X.txt'
#Y_TXT = 'sms_grouptime_Y.txt'
#VOCAB_TXT = 'sms_grouptime_vocab.txt'

#DATA_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/out/'
#X_TXT = 'all_se_source_X.txt'
#Y_TXT = 'all_se_source_Y.txt'
#VOCAB_TXT = 'vanilla_se.json_vocab.txt'

def main(_):
    tf.logging._logger.setLevel(logging.INFO)
    import os
    input_filename = os.path.join(DATA_DIR, X_TXT)
    output_filename = os.path.join(DATA_DIR, Y_TXT)
    vocab_filename = os.path.join(DATA_DIR, VOCAB_TXT)
    train_seq2seq(
        input_filename, output_filename, vocab_filename, MODEL_DIR)



if __name__ == '__main__':
    tf.app.run()
