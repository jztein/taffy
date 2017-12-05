import numpy as np
import tensorflow as tf

import data_prep

EMBED_DIM = 12

# from data_prep.py
START_TOKEN = 0
END_TOKEN = 1


def model_fn(features, labels, mode, params):
    input = features['input']
    output = features['output']  # TODO: Make this labels
    vocab_size = params['vocab_size']
    embed_dim = params['embedding_dim']
    cell_size = 64  # TODO: MAKE PARAM!
    batch_size = params['batch_size']
    output_len = params['output_len']

    input = tf.convert_to_tensor(input, name="yoloin")
    output = tf.convert_to_tensor(output, name="yoloout")
    print('inpUT', input)

    #embedded_input = tf.contrib.layers.embed_sequence(
    #    input, vocab_size=vocab_size, embed_dim=embed_dim,
    #    scope='embed', reuse=False)
    #embedded_output = tf.contrib.layers.embed_sequence(
    #    output, vocab_size=vocab_size, embed_dim=embed_dim,
    #    scope='embed', reuse=True)


    # Encoder.
    with tf.variable_scope('embedings'):
        embedding = tf.get_variable('embed_share', [vocab_size, cell_size])
    embedding_input = embedding
    embedding_output = embedding
    print('embeddding SHARE:', embedding)

    cell = tf.contrib.rnn.LSTMCell(num_units=cell_size)
    embedded_input = tf.nn.embedding_lookup(embedding_input, input)
    print('embeddding INPUT:', embedded_input)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        cell, embedded_input, dtype=tf.float32)

    start_tokens = tf.zeros([batch_size], dtype=tf.int32) # zeros since start = 0
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
    output_lengths = tf.reduce_sum(tf.to_int32(
            tf.not_equal(train_output, END_TOKEN)), 1)

    embedded_output = tf.nn.embedding_lookup(embedding_output, output)
    print('embeddding OUTPUT:', embedded_output)
    train_helper = tf.contrib.seq2seq.TrainingHelper(
        embedded_output, output_lengths, time_major=False)
    #pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    #    embeddings, start_tokens=tf.to_int32(start_tokens), end_token=END_TOKEN)

    # Decoder
    output_layer = tf.layers.Dense(
        vocab_size, use_bias=False, name="output_projection")
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input, END_TOKEN)), 1)
    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            #attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            #    num_units=cell_size, memory=encoder_outputs,
            #    memory_sequence_length=input_lengths)
            #attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            #    cell, attention_mechanism, attention_layer_size=cell_size / 2)
            a = '''
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                cell, vocab_size, reuse=reuse
                #attn_cell, vocab_size, reuse=reuse
            )#'''
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell, helper=helper,
                output_layer=output_layer,
                initial_state=encoder_final_state)
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                maximum_iterations=output_len,
            )
            print('############outputs', outputs)
            return outputs
    train_outputs = decode(train_helper, 'decode')
    #pred_outputs = decode(pred_helper, 'decode', reuse=True)

    weights = tf.to_float(tf.not_equal(train_output[:, :-1], END_TOKEN))
    loss = tf.contrib.seq2seq.sequence_loss(
        output_layer(train_outputs.rnn_output), output, weights=weights)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.001,
        summaries=['loss', 'learning_rate'])

    # TODO: Make dev, test
    #tf.identity(train_outputs.sample_id[0], name='training_output')
    #tf.identity(pred_outputs.sample_id[0], name='prediction_output')
    return tf.estimator.EstimatorSpec(
        mode=mode,
        #predictions=pred_outputs.sample_id,
        loss=loss,
        train_op=train_op
    )

def make_input_fn(in_ids, out_ids, ids_to_vocab):
    sample_in = [ids_to_vocab[i] for i in in_ids[0]]
    sample_out = [ids_to_vocab[i] for i in out_ids[0]]
    print('Sample:', sample_in, sample_out)

    for i, _ in enumerate(out_ids):
        out_ids[i].insert(0, START_TOKEN)
        del out_ids[i][-1]
        out_ids[i].append(END_TOKEN)

    def train_input_fn():
        return {'input': in_ids[1:5], 'output': out_ids[1:5]}, None

    return train_input_fn, sample_in, sample_out

def convert_ids_to_words(keys, ids_to_vocab):
    def format(key_ids):
        lines = []
        for key in keys:
            ids = key_ids[key]
            words = [ids_to_vocab[id] for id in ids]
            lines.append('%s: %s' % (key, ' '.join(words)))
        return '\n'.join(lines)

def main(_):
    in_ids, out_ids, vocab_to_ids, ids_to_vocab = data_prep.load_data_ids()
    print('In - out', len(in_ids), len(out_ids))

    params = {}
    params['vocab_size'] = len(ids_to_vocab)
    params['embedding_dim'] = EMBED_DIM
    #params['batch_size'] = len(in_ids)
    params['batch_size'] = 4
    params['output_len'] = len(in_ids[0])

    train_input_fn, sample_in, sample_out = make_input_fn(
        in_ids, out_ids, ids_to_vocab)

    #print_predictions = tf.train.LoggingTensorHook(
    #    ['prediction_output', 'training_output'], every_n_iter=100,
    #    formatter=convert_ids_to_words(
    #        ['prediction_output', 'training_output'], ids_to_vocab))

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=params, model_dir='/tmp/cs229/s2s/')
    estimator.train(
        input_fn=train_input_fn,
        steps=1000)#, hooks=[print_predictions])

    a = '''
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input': np.array(sample_in)},
        num_epochs=1,
        shuffle=False)
    print('Predictions:', estimator.predict(input_fn=predict_input_fn))
    pass'''


if __name__ == '__main__':
    tf.app.run()
