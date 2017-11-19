# Makes training data.
import json
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
    train = load_bow.loadAndConvertVanillaTrain(data_file)
    filename = makeFile(flags.train_json)
    with open(filename, 'w') as f:
        json.dump(train, f)


if __name__ == '__main__':
    if flags.data_type == 'kmeans':
        makeKmeansTrain(data_file)
    elif flags.data_type == 'vanilla':
        makeVanillaTrain(data_file)
