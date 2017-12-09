import collections
import os
import pickle

SUFFIX = '_1'

OUT_DIR = '/Users/kristenaw/Documents/Stanford/project/taffy/s2s/out/'
DATA_DIR = '/Users/kristenaw/Documents/Stanford/project/out/'
#OUT_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/taffy/s2s/out/'
#DATA_DIR = '/Users/jiayu/Documents/1Stanford/cs229/project/out/'

CMD_MK_VOCAB = 'mk_vocab'
CMD_MK_IDS = 'mk_converted_ids'

#cmd = CMD_MK_VOCAB
cmd = CMD_MK_IDS


START_TOKEN = "<GO>"  # id: 0
END_TOKEN = "<STOP>"   # id: 1
UNK_TOKEN = "<UNK>"
START_TOKEN_ID = 0
END_TOKEN_ID = 1
UNK_TOKEN_ID = 2


def make_vocab(filenames):
    word_counts = collections.Counter()
    for filename in filenames:
        with open(filename, 'r') as f:
            words = f.read()
            words = words.lower().split()
            word_counts.update(words)
    vocab = [START_TOKEN, END_TOKEN, UNK_TOKEN]
    vocab.extend([word for word in word_counts])
    vocab_to_ids = {word: id for id, word in enumerate(vocab)}
    id_to_vocabs = {id: word for word, id in vocab_to_ids.items()}
    return vocab_to_ids, id_to_vocabs


def save_vocab(data_filenames, vocab_prefix):
    vocab = make_vocab(data_filenames)
    vocab_filename = vocab_prefix + SUFFIX + '.pk'
    print('Pickling vocab to:', vocab_filename)
    with open(vocab_filename, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(vocab_prefix):
    # Returns vocab_to_ids, id_to_vocabs
    vocab_filename = vocab_prefix + SUFFIX + '.pk'
    with open(vocab_filename, 'rb') as f:
        return pickle.load(f)


def get_vocab_prefix():
    vocab_name = 'all_vocab' + SUFFIX
    vocab_name = os.path.join(OUT_DIR, vocab_name)
    return vocab_name


def get_data_filenames():
    input = 'all_se_source_X.txt'
    output = 'all_se_source_Y.txt'
    input_filename = os.path.join(DATA_DIR, input)
    output_filename = os.path.join(DATA_DIR, output)
    return [input_filename, output_filename]

def get_data_ids_filenames():
    input = 'all_se_source_X.txt' + SUFFIX + '.pk'
    output = 'all_se_source_Y.txt' + SUFFIX + '.pk'
    input_filename = os.path.join(DATA_DIR, input)
    output_filename = os.path.join(DATA_DIR, output)
    return [input_filename, output_filename]


def convert_to_ids(infile, vocab_to_ids):
    all_ids = []
    with open(infile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line: continue
            words = line.lower().split()
            ids = [vocab_to_ids[word] for word in words]
            all_ids.append(ids)
    return all_ids


def fix_ids_len(all_ids, fixed_len):
    for i, ids in enumerate(all_ids):
        all_ids[i] = ids + [END_TOKEN_ID] * (fixed_len - len(ids))
        all_ids[i][-1] = END_TOKEN_ID
    return all_ids

def save_ids(all_ids, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(all_ids, f)


def get_data(all_ids_file):
    # all_ids_file = convert_to_ids.outfile.
    with open(all_ids_file, 'rb') as f:
        return pickle.load(all_ids_file)


def convert_data_to_ids_and_save():
    vocab_name = get_vocab_prefix()
    vocab_to_ids, _ = load_vocab(vocab_name)
    data_in, data_out = get_data_filenames()
    in_ids = convert_to_ids(data_in, vocab_to_ids)
    out_ids = convert_to_ids(data_out, vocab_to_ids)

    total_len = 0
    num_lines = len(in_ids) + len(out_ids)
    for i, id in enumerate(in_ids):
        total_len += len(id)
        total_len += len(out_ids[i])
    fixed_len = int(total_len / num_lines) + 10
    in_ids = fix_ids_len(in_ids, fixed_len)

    out_ids = fix_ids_len(out_ids, fixed_len)
    ids_infile, ids_outfile = get_data_ids_filenames()
    save_ids(in_ids, ids_infile)
    save_ids(out_ids, ids_outfile)


def load_data_ids():
    vocab_name = get_vocab_prefix()
    vocab_to_ids, ids_to_vocab = load_vocab(vocab_name)
    ids_in, ids_out = get_data_ids_filenames()

    with open(ids_in, 'rb') as f_in:
        with open(ids_out, 'rb') as f_out:
            return pickle.load(f_in), pickle.load(f_out), vocab_to_ids, ids_to_vocab


if __name__ == '__main__':
    vocab_name = get_vocab_prefix()
    data_filenames = get_data_filenames()

    if cmd == CMD_MK_VOCAB:
        save_vocab(data_filenames, vocab_name)

    elif cmd == CMD_MK_IDS:
        convert_data_to_ids_and_save()
