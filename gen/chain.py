import os
import pickle
import random

import numpy as np
import scipy.sparse

VOCAB_DIR = '/Users/kristenaw/Documents/Stanford/project/taffy/s2s/out/'
X_Y_DIR = '/Users/kristenaw/Documents/Stanford/project/out/'
SUFFIX = '_1'
# Names lifted from data_prep.py
X_FILE = 'all_se_source_X.txt' + SUFFIX + '.pk'
Y_FILE = 'all_se_source_Y.txt' + SUFFIX + '.pk'
VOCAB_FILE = 'all_vocab' + SUFFIX + SUFFIX + '.pk'


def unpickle(dir, filename):
    filepath = os.path.join(dir, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


x_ids = unpickle(X_Y_DIR, X_FILE)
y_ids = unpickle(X_Y_DIR, Y_FILE)
print('Num X lines', len(x_ids))
print('Num Y lines', len(y_ids))

vocab_to_ids, ids_to_vocab = unpickle(VOCAB_DIR, VOCAB_FILE)
IDS = ids_to_vocab.keys()
STOP = None
for id, vocab in ids_to_vocab.iteritems():
    if vocab == '.':
        STOP = id
        print('STOP id:', STOP)
    elif vocab == 'though??':
        print('though??: %s' % id)
if STOP is None:
    print('STOP is none')
    STOP = len(vocab_to_ids)
    vocab_to_ids['.'] = STOP
    ids_to_vocab[STOP] = '.'

num_vocab = len(vocab_to_ids)
print('NUM vocab:', num_vocab)


NUM_AB = num_vocab * 10 + num_vocab
NUM_C = num_vocab
AB_RANGE = range(NUM_AB)
C_RANGE = range(NUM_C)


def ps_idx(word_i, word_j):
    """Makes index for PS."""
    return word_i * 10 + word_j


class BaseChain(object):

    def __init__(self):
        num_ab = NUM_C if self.use_2 else NUM_AB

        # PS[ps_idx(a, b)][c] = probability
        self.PS = np.zeros((num_ab, NUM_C))
        # PS's numerator.
        self.ABC = np.zeros((num_ab, NUM_C))
        # PS's denominator.
        self.AB = np.zeros(num_ab)

    def update(self, lines):
        raise ValueError('update not implemented %s' % self.__name__)

    def get_best_c(self, a, b):
        if self.use_2:
            c_probs = self.PS[a]
        else:
            c_probs = self.PS[ps_idx(a, b)]

        if np.sum(c_probs) == 0.:
            return None
        if True:
            total_probs = np.sum(c_probs)
            prob = random.uniform(0., total_probs)
            for c in C_RANGE:
                prob -= c_probs[c]
                if prob <= 0.:
                    best_c = c
                    break
            return best_c
        best_c = np.argmax(c_probs)
        return best_c

    def new_ab(self):
        a = IDS[-1]
        b = None
        while a == IDS[-1]:
            a = random.choice(IDS)
            if not self.use_2:
                b = a + 1
        return a, b


class TwoChain(BaseChain):

    def __init__(self):
        self.use_2 = True
        super(TwoChain, self).__init__()

    def update(self, lines):
        num_processed = 0
        for line in lines:
            if len(line) < 2:
                continue
            num_processed += 1
        
            for i in xrange(len(line) - 1):
                a = line[i]
                c = line[i + 1]
                #c = line[i + 2]
                ab_i = a#ps_idx(a, b)
                self.AB[ab_i] += 1 
                self.ABC[ab_i, c] += 1 
                self.PS[ab_i, c] = float(self.ABC[ab_i, c]) / self.AB[ab_i]

        print('Two chain processed:', num_processed)


class ThreeChain(BaseChain):

    def __init__(self):
        self.use_2 = False
        super(ThreeChain, self).__init__()

    def update(self, lines):
        num_processed = 0
        for line in lines:
            if len(line) < 3:
                continue
            num_processed += 1
            for i in xrange(len(line) - 2):
                a = line[i]
                b = line[i + 1]
                c = line[i + 2]
                ab_i = ps_idx(a, b)
                self.AB[ab_i] += 1 
                self.ABC[ab_i, c] += 1 
                self.PS[ab_i, c] = float(self.ABC[ab_i, c]) / self.AB[ab_i]
        print('Three chain processed:', num_processed)

    def update_with_reply(self, x_lines, y_lines):
        num_processed = 0
        for k, x_line in enumerate(x_lines):
            if len(x_line) < 2:
                continue
            num_processed += 1
            for i in xrange(len(x_line) - 1):
                a = x_line[i]
                b = x_line[i + 1]
                ab_i = ps_idx(a, b)
                self.AB[ab_i] += 1 

                for j, y_c in enumerate(y_lines[k]):
                    self.ABC[ab_i, y_c] += 1 * (len(y_lines[k]) - j)
                    self.PS[ab_i, y_c] = float(
                        self.ABC[ab_i, y_c]) / self.AB[ab_i]
        print('Three chain with reply processed:', num_processed)


def make_chain(ChainClazz, use_2, use_reply=False):
    print('Making chain:', ChainClazz.__name__, use_reply)
    suffix = '_n2' if use_2 else '_n3'
    if use_reply:
        suffix += '_reply'

    chain = ChainClazz()
    if use_reply:
        chain.update_with_reply(x_ids, y_ids)
    else:
        chain.update(x_ids)
        chain.update(y_ids)
    s_PS = scipy.sparse.csr_matrix(chain.PS)
    s_AB = scipy.sparse.csr_matrix(chain.AB)
    s_ABC = scipy.sparse.csr_matrix(chain.ABC)

    scipy.sparse.save_npz('PS' + suffix + '.npz', s_PS)
    scipy.sparse.save_npz('AB' + suffix + '.npz', s_AB)
    scipy.sparse.save_npz('ABC' + suffix + '.npz', s_ABC)

    print('saved chain!\n')
    return chain


def make_dense(file):
    sparse = scipy.sparse.load_npz(file)
    return sparse.toarray()
a = '''
    dense = np.zeros(sparse.shape)
    rows = sum((m * [k] for k, m in enumerate(np.diff(sparse.indptr))), [])
    dense[rows, sparse.indices] = sparse.data
    return dense'''


def load_chain(ChainClazz, use_2, use_reply=False):
    suffix = '_n2' if use_2 else '_n3'
    if use_reply:
        suffix += '_reply'
    PS = make_dense('PS' + suffix + '.npz')
    AB = make_dense('AB' + suffix + '.npz')
    ABC = make_dense('ABC' + suffix + '.npz')
    chain = ChainClazz()
    chain.PS = PS
    chain.AB = AB
    chain.ABC = ABC
    print(np.sum(PS))
    print(np.sum(AB))
    print(np.sum(ABC))

    return chain

def main():
    #random.seed()
    #with_reply = True
    with_reply = False
    if with_reply:
        reply_chain()
        return

    combined = True
    #combined = False
    if combined:
        two_chainz()
    else:
        one_chain(use_2=False)


def reply_chain():
    #fresh_chain = True
    fresh_chain = False
    if fresh_chain:
        chain = make_chain(ThreeChain, False, True)
        chain2 = make_chain(TwoChain, True, False)
    else:
        chain = load_chain(ThreeChain, False, True)
        chain2 = load_chain(TwoChain, True, False)
    for _ in xrange(10):
        generate_one_two_chainz_line(chain2, chain, init_with_3=True)



def two_chainz():
    #fresh_chain = True
    fresh_chain = False
    if fresh_chain:
        chain2 = make_chain(TwoChain, True)
        chain3 = make_chain(ThreeChain, False)
    else:
        chain2 = load_chain(TwoChain, True)
        chain3 = load_chain(ThreeChain, False)
    for _ in xrange(10):
        #generate_two_chainz_line(chain2, chain3)
        generate_one_two_chainz_line(chain2, chain3)


def generate_two_chainz_line(chain2, chain3, max_len=8):
    w1, w2 = chain3.new_ab()
    words = [w1, w2]
    while len(words) < max_len:
        c = chain3.get_best_c(w1, w2)

        if c is None:
            c = chain2.get_best_c(w1, None)
            if c is None:
                c = chain2.get_best_c(w2, None)
                if c is None:
                    c, _ = chain2.new_ab()

        words.append(c)
        w1, w2 = w2, c

    line = [ids_to_vocab.get(w, '^') for w in words]
    print('> %s' % ' '.join(line))


def generate_one_two_chainz_line(chain2, chain3, max_len=8, init_with_3=False):
    if init_with_3:
        w0, w1 = chain3.new_ab()
        words = [w0, w1]
        max_len=10
    else:
        w1, _ = chain2.new_ab()
        w0 = None
        words = [w1]
    while len(words) < max_len:
        c = chain2.get_best_c(w1, None)

        if c is None:
            # break_early
            #if True: break
            if w0 is not None:
                c = chain3.get_best_c(w0, w1)
            if c is None:
                c, _ = chain2.new_ab()

        words.append(c)
        w0 = w1
        w1 = c

    line = [ids_to_vocab.get(w, '^') for w in words]
    print('> %s' % ' '.join(line))



def one_chain(use_2):
    #fresh_chain = True
    fresh_chain = False
    #random.seed()

    chain_clazz = TwoChain if use_2 else ThreeChain
    chain = make_chain(chain_clazz, use_2) if (
        fresh_chain) else load_chain(chain_clazz, use_2)

    for _ in xrange(10):
        generate_line(chain, use_2)

def generate_line(chain, use_2, max_len=8):
    words = []

    if use_2:
        w1, _ = chain.new_ab()
        words = [w1]
        seen = set()
        while len(words) < max_len:
            c = chain.get_best_c(w1, None)
            if (w1, c) not in seen:
                seen.add((w1, c))
            else:  # Choose again!
                w1, _ = chain.new_ab()
                continue

            if c is None:
                # break_early
                #if True: break
                w1, _ = chain.new_ab()
                continue

            words.append(c)
            w1 = c

    else:
        w1, w2 = chain.new_ab()
        words = [w1, w2]
        while len(words) < max_len:
            c = chain.get_best_c(w1, w2)

            if c is None:
                w1, w2 = chain.new_ab()
                continue

            words.append(c)
            w1, w2 = w2, c

    line = [ids_to_vocab.get(w, '^') for w in words]
    print('> %s' % ' '.join(line))


if __name__ == '__main__':
    main()



