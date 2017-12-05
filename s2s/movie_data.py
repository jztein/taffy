# Prepares movie lines.
import collections
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
    return line_map, ['<GO>', '<STOP>', '<UNK>'] + vocab.keys()


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
    return X, Y


if __name__ == '__main__':
    line_map, vocab = load_lines_and_vocab(lines_file)
    X, Y = make_convs(conversations_file, line_map)

    x_file = os.path.join(OUT_DIR, 'movie_lines_X.txt')
    y_file = os.path.join(OUT_DIR, 'movie_lines_Y.txt')
    vocab_file = os.path.join(OUT_DIR, 'movie_lines_vocab.txt')


    with open(x_file, 'w') as xf:
        with open(y_file, 'w') as yf:
            with open(vocab_file, 'w') as vf:
                xf.write('\n'.join(X))
                yf.write('\n'.join(Y))
                vf.write('\n'.join(vocab))
