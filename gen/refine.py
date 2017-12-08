import pickle
import random
import numpy as np


SRC1 = 'what'
TGT1 = 'wat'

SRC_TGTS = [
    ('what', 'wat'),
    ('what', 'whatt'),
    ('what', 'wassup'),
    ('you', 'u'),
    ('tomorrow', 'tmr'),
    ('hello', 'hey'),
    ('hi', 'hey'),
    ('okay', 'k'),
    ('okay', 'ok'),
]

# TODO: emoticons.
# NOTE: wan to keep spaces, punctuation

#  learn policy of letter -> changed letter 
# insert = letter -> letter + letter2
# subtract = letter -> -2
# repl = letter -> letter2
# noop = letter -> letter  ## needed?


alphabet = 'abcdefghijklmnopqrstuvwxyz '

alpha_ids = {letter: i for i, letter in enumerate(alphabet)}
id_alphas = {i: letter for letter, i in alpha_ids.items()}

IDS = id_alphas.keys()
BLANK = IDS[-1]

INSERT = 0
SUBTRACT = 1
REPLACE = 2
NOOP = 3
ACTIONS = [INSERT, SUBTRACT, REPLACE, NOOP]

# a state: (idx of initial letter, idx of changed letter)
STATES = [(i, j) for i in IDS for j in IDS]
NUM_STATES = len(STATES)
print('|S|:', NUM_STATES)

# To access: SA[i][j][a].
SA = [[[0 for a in ACTIONS] for j in IDS] for i in IDS]
# To access: SA_S2[i][j][a][i2][j2].
SA_S2 = [[[[[0 for j2 in IDS] for i2 in IDS] for a in ACTIONS] for j in IDS] for i in IDS]
# PSA(S2) = SA_S2 / SA.
DTP = 1. / NUM_STATES # DEFAULT_TRANSITION_PROBABILITY.
PSA = [[[[[DTP for j2 in IDS] for i2 in IDS] for a in ACTIONS] for j in IDS] for i in IDS]

VALUE = [[random.uniform(0., .1) for j in IDS] for i in IDS]

REWARD = [[0. for j in IDS] for i in IDS]
MIN_REWARD = -1000.

GAMMA = 0.997
TOLERANCE = 0.01

def to_ids(line):
    ids = []
    for letter in line:
        ids.append(alpha_ids.get(letter, BLANK))
    return ids


def to_line(ids):
    line = []
    for id in ids:
        line.append(id_alphas.get(id, ' '))
    return ''.join(line)

def psa(s, a, s2):
    return PSA[s[0]][s[1]][a][s2[0]][s2[1]]

def update_psa(s, a, s2, value):
    PSA[s[0]][s[1]][a][s2[0]][s2[1]] = value

def value(s):
    return VALUE[s[0]][s[1]]

def set_value(s, val):
    VALUE[s[0]][s[1]] = val

def sa(s, a):
    return SA[s[0]][s[1]][a]

def incr_sa(s, a):
    SA[s[0]][s[1]][a] += 1

def sa_s2(s, a, s2):
    return SA_S2[s[0]][s[1]][a][s2[0]][s2[1]]

def incr_sa_s2(s, a, s2):
    SA_S2[s[0]][s[1]][a][s2[0]][s2[1]] += 1

def get_reward(s):
    return REWARD[s[0]][s[1]]

def set_reward(next_state, rough, target):
    REWARD[next_state[0]][next_state[1]] = reward(rough, target)    

def reset_reward(next_state):
    REWARD[next_state[0]][next_state[1]] = 0


def reward(h, y):
    m = 1 # TODO: Make this a batch pertub?
    max_len = max(len(h), len(y))
    min_len = min(len(h), len(y))
    cost = float(max_len - min_len)
    for i in range(min_len):
        if h[i] != y[i]:
            cost += 1.
    r = max(-cost, MIN_REWARD)
    return r


def policy(state):
    """Returns policy's suggested action."""
    best_action = []
    best_ev = -1000.
    for action in ACTIONS:
        ev = 0.
        for state2 in STATES:
            ev += psa(state, action, state2) * value(state2)
        if ev > best_ev:
            best_action = [action]
            best_ev = ev
        elif ev == best_ev:
            best_action.append(action)
    #print('best action: %s, %s' % (best_action, best_ev))
    if not best_action:
        print('no best action')
        return random.choice(ACTIONS)
    return random.choice(best_action)


def refine(state, rough, target):
    """Rough is line of IDS."""
    cur_letter = BLANK
    while cur_letter == BLANK:
        idx = random.choice(range(len(rough)))
        cur_letter = rough[idx]

    if state is None:
        state = (cur_letter, cur_letter)

    action = policy(state)
    best_letter = random.choice(IDS)  # best_letter = best_i.

    if action == INSERT:
        rough.insert(idx, best_letter)
        next_state = (cur_letter, best_letter)
    elif action == SUBTRACT:
        del rough[idx]
        next_state = (cur_letter, BLANK)
    elif action == REPLACE:
        rough[idx] = best_letter
        next_state = (cur_letter, best_letter)
    else:  # action == NOOP.
        next_state = (cur_letter, cur_letter)

    incr_sa(state, action)
    incr_sa_s2(state, action, next_state)
    
    set_reward(next_state, rough, target)

    state = next_state

    return state, rough



def change(a_line, b_line):
    print('original: %s\ntarget: %s' % (a_line, b_line))

    a_ids, b_ids = to_ids(a_line), to_ids(b_line)
    state = None

    NO_LEARNING_THRESHOLD = 10
    consecutive_no_learning_trials = 1
    last_converged_in_one_iteration = False

    max_dissimilarity = - (max(len(a_ids), len(b_ids)) / 2)

    steps = 0
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:
        steps += 1

        state, a_ids = refine(state, a_ids, b_ids)

        similarity = reward(a_ids, b_ids)
        got_best = similarity == 0

        if similarity < max_dissimilarity:
            # Update PSA.
            for s in STATES:
                for a in ACTIONS:
                    for s2 in STATES:
                        if sa(s, a) > 0:
                            prob = float(sa_s2(s, a, s2) / sa(s, a))
                            #if prob > 0.:
                            #    print('new prob', s, a, s2, prob, sa(s, a))
                            update_psa(s, a, s2, prob)

            # Update policy: V.
            num_iters = 0
            while True:
                num_iters += 1
                new_Vs = 0.
                old_Vs = 0.
                if num_iters > 100: break
            
                for s in STATES:
                    immediate_reward = get_reward(s)
                    
                    best_fev = 0.
                    for a in ACTIONS:
                        fev = GAMMA * sum([psa(s, a, s2) * value(s2)
                                           for s2 in STATES])
                        if fev > best_fev:  # max_a.
                            best_fev = fev
                    old_V = value(s)
                    set_value(s, immediate_reward + best_fev)
                    old_Vs += old_V 
                    new_Vs += value(s)
                    #print('Old new V[s]:',
                    #      old_Vs, new_Vs, immediate_reward, best_fev)
                # Check if converged
                if abs(new_Vs - old_Vs) < TOLERANCE:
                    print (num_iters, '::S:', s, ': ', old_Vs, '->', new_Vs)
                    print('refined: %s' % to_line(a_ids))
                    if num_iters == 1:
                        print ('S:', s, ': ', old_Vs, '->', new_Vs)
                        if last_converged_in_one_iteration:
                            consecutive_no_learning_trials += 1
                        else:
                            consecutive_no_learning_trials = 1
                            last_converged_in_one_iteration = True

                    break

            if got_best:
                break

        # Reset.
        if similarity < max_dissimilarity:
            a_ids = to_ids(a_line)


    print('in %s steps: : final: %s\ntarget: %s' % (steps, to_line(a_ids), b_line))


if __name__ == '__main__':
    for src, tgt in SRC_TGTS:
        change(src, tgt)

    with open('refine_1.pkl', 'wb') as f:
        pickle.dump((PSA, VALUE, REWARD, SA, SA_S2), f)
