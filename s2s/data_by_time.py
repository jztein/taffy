# Groups SMSes by conversations in threads.
import collections
import json
import os

OUT_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/taffy/s2s/out/'
DATA_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/out/'

def load_data(filename):
    with open(filename, 'r') as f:
        return json.loads(f.read())

# SMS time is in ms.
HOURS_24 = 24 * 60 * 60 * 1000

def group_by_time(data, x_filename, y_filename, vocab_filename):
    X = []
    Y = []
    vocab = collections.Counter()
    for thread in data.values():
        last_time = 0
        conversation = []
        for msg in thread['thread']:
            cur_time = int(msg['date'])
            if cur_time == 0 or cur_time - last_time < HOURS_24:
                msg_words = msg['msg'].lower().encode('utf-8').split()
                vocab.update(msg_words)
                msg_text = (' '.join(msg_words))

                if conversation:
                    X.append(' '.join(conversation))
                    Y.append(msg_text)

                conversation.append(msg_text)
            else:
                conversation = []

            last_time = cur_time


    with open(x_filename, 'w') as f:
        f.write('\n'.join(X))
    with open(y_filename, 'w') as f:
        f.write('\n'.join(Y))
    with open(vocab_filename, 'w') as f:
        f.write('\n'.join(['<GO>', '<STOP>', '<UNK>'] + vocab.keys()))


if __name__ == '__main__':
    data_filename = os.path.join(DATA_DIR, 'sms-20171118000041.xml_se.json')
    x_filename = os.path.join(OUT_DIR, 'sms_grouptime_X.txt')
    y_filename = os.path.join(OUT_DIR, 'sms_grouptime_Y.txt')
    vocab_filename = os.path.join(OUT_DIR, 'sms_grouptime_vocab.txt')

    group_by_time(
        load_data(data_filename), x_filename, y_filename, vocab_filename)
