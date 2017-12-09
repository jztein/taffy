import os
import pickle
import random

import numpy as np

VOCAB_DIR = '/Users/kristenaw/Documents/Stanford/project/taffy/s2s/out/'
X_Y_DIR = '/Users/kristenaw/Documents/Stanford/project/out/'
SUFFIX = '_1'
# Names lifted from data_prep.py                                                
X_FILE = 'all_se_source_X.txt' + SUFFIX + '.pk'
Y_FILE = 'all_se_source_Y.txt' + SUFFIX + '.pk'
VOCAB_FILE = 'all_vocab' + SUFFIX + SUFFIX + '.pk'

WAPP_FILES = ['cwn.txt', 'cxy.txt', 'sc.txt', 'wn.txt']


def unpickle(dir, filename):
    filepath = os.path.join(dir, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


vocab_to_ids, ids_to_vocab = unpickle(VOCAB_DIR, VOCAB_FILE)
x_ids = unpickle(X_Y_DIR, X_FILE)
y_ids = unpickle(X_Y_DIR, Y_FILE)
IDS = ids_to_vocab.keys()

wa_x_lines = []
wa_y_lines = []


def read_data(wapp_file, x_lines, y_lines):
    global x_ids, y_ids
    global wa_x_lines, wa_y_lines

    last_line = None
    last_ids = None
    with open(wapp_file, 'r') as f:
        for line in f.readlines():
            if 'Messages to this chat' in line or '<Media omitted>' in line:
                continue

            # index of second : is end of `<date>, <time> - <name>:` prefix.
            second_colon = line.find(':', 13)
            words = line[second_colon+1:].strip().lower().split()

            if last_line is None:
                last_line = ' '.join(words)
            else:
                cur_line = ' '.join(words)
                x_lines.append(last_line)
                y_lines.append(cur_line)
                wa_x_lines.append(last_line)
                wa_y_lines.append(cur_line)
                last_line = cur_line

            ids = []
            for word in words:
                if word in vocab_to_ids:
                    ids.append(vocab_to_ids[word])
                else:
                    word_id = IDS[-1] + 1
                    IDS.append(word_id)
                    ids_to_vocab[word_id] = word
                    vocab_to_ids[word] = word_id
                    ids.append(word_id)

            if last_ids is None:
                last_ids = ids
                continue

            x_ids.append(last_ids)
            y_ids.append(ids)
            last_ids = ids


if __name__ == '__main__':
    x_lines, y_lines = [], []
    for wapp_file in WAPP_FILES:
        read_data(wapp_file, x_lines, y_lines)

    print('new x y lines (%s, %s)' % (len(x_ids), len(y_ids)))

    prefix = 'wapp_'
    with open(prefix + X_FILE, 'wb') as f:
        pickle.dump(x_ids, f)
    with open(prefix + Y_FILE, 'wb') as f:
        pickle.dump(y_ids, f)
    with open(prefix + VOCAB_FILE, 'wb') as f:
        pickle.dump((vocab_to_ids, ids_to_vocab), f)
    with open(prefix + 'lines_x.txt', 'w') as f:
        f.write('\n'.join(x_lines))
    with open(prefix + 'lines_y.txt', 'w') as f:
        f.write('\n'.join(y_lines))
    with open(prefix + 'wa_only_lines_x.txt', 'w') as f:
        f.write('\n'.join(x_lines))
    with open(prefix + 'wa_only_lines_y.txt', 'w') as f:
        f.write('\n'.join(y_lines))
