# Makes training data.
from __future__ import print_function
import json
import random
import os
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer

import load_bow


flag_parser = argparse.ArgumentParser()
flag_parser.add_argument('data_type')  # vanilla, kmeans
flag_parser.add_argument('outdir')
flag_parser.add_argument('train_json')
flags = flag_parser.parse_args()

def makeFile(name):
    return os.path.join(flags.outdir, name)

if flags.data_type == 'kmeans':
    idx_map_json = makeFile('idx_map_kmeans_n11.json')
    label_map_json = makeFile('label_map_kmeans_n11.json')
    with open(idx_map_json, 'r') as f:
        idx_map = json.load(f)

    with open(label_map_json, 'r') as f:
        label_map = json.load(f)

data_file = makeFile('sms-20171118000041.xml_se.json')

SCHEDULE = 10
ACTIVITIES = 1
SGTM = 0
RESPONSES = 4
IDENTITY = 7
FEEDBACK = 6

CLUSTERS = [SCHEDULE, ACTIVITIES, SGTM, RESPONSES, IDENTITY, FEEDBACK]

def makeKmeansTrain(data_file):
    data = load_bow.loadData(data_file)
    cluster_map = {str(c): [] for c in CLUSTERS}

    for c_i in CLUSTERS:
        c_i = str(c_i)
        samples = idx_map[c_i]
        for idx in samples:
            if idx+1 >= len(data):
                continue
            cluster_map[c_i].append({'x': data[idx], 'y': data[idx+1]})

    filename = makeFile(flags.train_json)
    with open(filename, 'w') as f:
        json.dump(cluster_map, f)


def makeVanillaTrain(data_file):
    data_json = load_bow.loadAndConvertVanillaTrain(data_file)
    filename = makeFile(flags.train_json)
    with open(filename, 'w') as f:
        json.dump(data_json, f)

    indices = [i for i in xrange(len(data_json))]
    random.shuffle(indices)
    eighty_p = int(.8 * len(data_json))
    ninety_p = int(.9 * len(data_json))
    train_json, dev_json, test_json = [], [], []
    for i, index in enumerate(indices):
        if i < eighty_p:
            train_json.append(data_json[index])
        elif i < ninety_p:
            dev_json.append(data_json[index])
        else:
            test_json.append(data_json[index])

    outfile = flags.train_json
    makeDataFile(makeFile('train_' + outfile), train_json)
    makeDataFile(makeFile('dev_' + outfile), dev_json)
    makeDataFile(makeFile('test_' + outfile), test_json)

    with open(filename + '_vocab.txt', 'w') as fv:
        vocab = load_bow.loadVanillaTrainVocab(data_json)
        for word in vocab:
            print(word, file=fv)

def makeDataFile(filename, data_json):
    with open(filename + '_X.txt', 'w') as fx:
        with open(filename + '_Y.txt', 'w') as fy:
            for xy in data_json:
                # encode else will get ascii encoding UnicodeEncodeError's.
                print(xy['x'], file=fx)
                print(xy['y'], file=fy)


if __name__ == '__main__':
    if flags.data_type == 'kmeans':
        makeKmeansTrain(data_file)
    elif flags.data_type == 'vanilla':
        makeVanillaTrain(data_file)
    elif flags.data_type == 'all':
        data_json = load_bow.loadAndConvertVanillaTrain(data_file)
        makeDataFile(makeFile('all_' + flags.train_json), data_json)
