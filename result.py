# parses kmeans results (looks at the clusters)
import json
import numpy as np
import argparse

fp = argparse.ArgumentParser()
fp.add_argument('data')



def loadData(infile):
    with open(infile, 'r') as f:
        indata = json.load(f)
    data = []
    for thread in indata.itervalues():
        for msg in thread['thread']:
            data.append(msg['msg'])
    return data


data = loadData('../out/sms-20171118000041.xml_se.json')

labels = np.load('X_labels.npy')

idx_map = {}
label_map = {}
for i, label in enumerate(labels):
    label = str(label)
    if label not in label_map:
        label_map[label] = []
        idx_map[label] = []
    label_map[label].append(data[i])
    idx_map[label].append(i)

print 'num_results:', len(label_map)
for things in idx_map.itervalues():
    print 'c count:', len(things)

if True:
    with open('../out/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=4, separators=(',', ':'))
    with open('../out/idx_map.json', 'w') as f:
        json.dump(idx_map, f, indent=4, separators=(',', ':'))
