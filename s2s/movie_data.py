# Prepares movie lines.
import collections
import random
import re
import os

# CSV separator
SEP = ' +++$+++ '

OUT_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/taffy/s2s/out/'
DATA_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/data/cornell movie-dialogs corpus/'

LINES = 'movie_lines.txt'
CONVS = 'movie_conversations.txt'


lines_file = os.path.join(DATA_DIR, LINES)
conversations_file = os.path.join(DATA_DIR, CONVS)

def load_lines_and_vocab(filename):
    line_map = {}
    vocab = collections.Counter()
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_id, u0, m0, n0, speech = line.split(SEP)
            try:
                speech_words = speech.decode('utf-8', 'ignore').lower().split()
            except UnicodeDecodeError:
                print('Undecodable:', speech)
                continue
            line_map[line_id] = ' '.join(speech_words)
            vocab.update(speech_words)
    return line_map, ['<unk>', '<s>', '</s>'] + vocab.keys()


def make_convs(filename, line_map):
    X = []
    Y = []
    with open(filename, 'r') as f:
        for conv in f.readlines():
            u0, u2, m0, line_ids = conv.split(SEP)
            line_ids = re.sub("[\[\]'\\n]", '', line_ids)
            line_ids = [id.strip() for id in line_ids.split(',')]
            for i in range(len(line_ids) - 1):
                j = i + 1
                x = line_map.get(line_ids[i])
                y = line_map.get(line_ids[j])
                if x is None or y is None: continue
                X.append(x)
                Y.append(y)
    random.shuffle(X)
    random.shuffle(Y)
    return X, Y


if __name__ == '__main__':
    line_map, vocab = load_lines_and_vocab(lines_file)
    X, Y = make_convs(conversations_file, line_map)

    x_file = os.path.join(OUT_DIR, 'movie_lines')
    y_file = os.path.join(OUT_DIR, 'movie_lines')
    vocab_file = os.path.join(OUT_DIR, 'movie_lines_vocab')

    x_suffix = '.x'
    y_suffix = '.y'

    with open(vocab_file + x_suffix, 'w') as vfx:
        with open(vocab_file + y_suffix, 'w') as vfy:
            vfx.write('\n'.join(vocab))
            vfy.write('\n'.join(vocab))

    num_train = int(len(X)*.8)
    num_test = int(len(X)*.1)
    train_X, train_Y = X[:num_train], Y[:num_train]
    dev_X, dev_Y = X[num_train:-num_test], Y[num_train:-num_test]
    test_X, test_Y = X[-num_test:], Y[-num_test:]

    with open(x_file + '_dev' + x_suffix, 'w') as xf:
        with open(y_file + '_dev' + y_suffix, 'w') as yf:
            xf.write('\n'.join(dev_X))
            yf.write('\n'.join(dev_Y))

    with open(x_file + '_train' + x_suffix, 'w') as xf:
        with open(y_file + '_train' + y_suffix, 'w') as yf:
            xf.write('\n'.join(train_X))
            yf.write('\n'.join(train_Y))

    with open(x_file + '_test' + x_suffix, 'w') as xf:
        with open(y_file + '_test' + y_suffix, 'w') as yf:
            xf.write('\n'.join(test_X))
            yf.write('\n'.join(test_Y))
