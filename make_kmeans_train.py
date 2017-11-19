# Makes training data found from kmeans clustering.
import json
import os
import argparse

import load_bow


flag_parser = argparse.ArgumentParser()
flag_parser.add_argument('outdir')
flag_parser.add_argument('cluster_json')
flags = flag_parser.parse_args()

def makeFile(name):
    return os.path.join(flags.outdir, name)

idx_map_json = makeFile('idx_map_kmeans_n11.json')
label_map_json = makeFile('label_map_kmeans_n11.json')
bow_file = makeFile('sms-20171118000041.xml_se.json')

SCHEDULE = 10
ACTIVITIES = 1
SGTM = 0
RESPONSES = 4
IDENTITY = 7
FEEDBACK = 6

CLUSTERS = [SCHEDULE, ACTIVITIES, SGTM, RESPONSES, IDENTITY, FEEDBACK]

with open(idx_map_json, 'r') as f:
    idx_map = json.load(f)

with open(label_map_json, 'r') as f:
    label_map = json.load(f)

data = load_bow.loadData(bow_file)

cluster_map = {str(c): {'xs': [], 'ys': []} for c in CLUSTERS}

for c_i in CLUSTERS:
    c_i = str(c_i)
    samples = idx_map[c_i]
    for idx in samples:
        if idx+1 >= len(data):
            continue
        cluster_map[c_i]['xs'].append(data[idx])
        cluster_map[c_i]['ys'].append(data[idx+1])

filename = makeFile(flags.cluster_json)
with open(filename, 'w') as f:
    json.dump(cluster_map, f)
