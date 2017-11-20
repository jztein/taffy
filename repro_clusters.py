"""Kmeans-based predictions and eval.

$ python repro_clusters.py ../out all_se_source_X.txt all_se_source_Y.txt X_label_map_train1.pk km_centroids_train1.npy X_train1.npz kmeans_dataset_train1.pk
"""
import argparse
import pickle
import random

import numpy as np
from scipy.sparse import load_npz

import load_bow

flag_parser = argparse.ArgumentParser()
flag_parser.add_argument('outdir')
flag_parser.add_argument('src')
flag_parser.add_argument('tgt')
flag_parser.add_argument('label_map')
flag_parser.add_argument('centroids')
flag_parser.add_argument('x_npz')
flag_parser.add_argument('dataset')
flag_parser.add_argument('--repro')
flags = flag_parser.parse_args()

SE_DATA = '../out/sms-20171118000041.xml_se.json'
LABELS = '../out/X_labels_train1.npy'
CENTROIDS = '../out/X_centroids_train1.npy'

def getXAndDataset(mode, x_npz_file, dataset_file):
    train, dev, test = pickle.load(open(dataset_file, 'rb'))
    X = load_npz(x_npz_file)
    if mode == 'train':
        return X, train
    if mode == 'dev':
        return X, dev
    if mode == 'test':
        return X, test


def reproCentroids(km_labels, centroids_file):
    data = load_bow.loadData(SE_DATA)
    X, vectorizer = load_bow.vectorize(data)

    clusters = [[] for _ in xrange(11)]

    for i, c_i in enumerate(km_labels):
        clusters[c_i].append(X[i].todense())

    centroids = [None for _ in xrange(11)]

    for c_i, cluster in enumerate(clusters):
        centroids[c_i] = np.average(cluster, axis=0)

    print 'centroids:', centroids

    np.save(centroids_file, centroids)

    return centroids


def loadCentroids(centroids_file):
    return np.load(centroids_file)


def predictCluster(centroids, src_vec):
    min_dist = 11111111111110
    best_c_i = 0
    for i, centroid in enumerate(centroids):
        cur_dist = np.linalg.norm(centroid - src_vec)

        if cur_dist < min_dist:
            min_dist = cur_dist
            best_c_i = i
    return best_c_i


def eval(centroids, src_file, tgt_file, label_map, X, X_indices):
    to_ignore = []
    for label in label_map:
        print 'Size:', label, len(label_map[label])
        if len(label_map[label]) < 10:
            to_ignore.append(label)
    print 'to ignore:', to_ignore

    results = []

    with open(src_file, 'r') as fs:
        with open(tgt_file, 'r') as ft:
            srcs = fs.readlines()
            tgts = ft.readlines()
            print 'here', len(srcs), len(tgts)
            for x_i in X_indices:
                if x_i > len(srcs) or x_i > len(tgts):
                    print 'More:', x_i, len(srcs), len(tgts)
                    break

                src = srcs[x_i]
                tgt = tgts[x_i]
                src_vec = X[x_i]

                best_c_i = predictCluster(centroids, src_vec)
                h_i = random.choice(label_map[str(best_c_i)])
                h = tgts[h_i]

                results.append('----')
                results.append(src)
                results.append(tgt)
                results.append(h)
                results.append(str(best_c_i))

    with open('kmeans_eval_train1.txt', 'w') as f:
        f.write('\n'.join(results))


def main():
    labels_file = load_bow.makeFile(flags.outdir, flags.label_map)
    centroids_file = load_bow.makeFile(flags.outdir, flags.centroids)
    srcs_file = load_bow.makeFile(flags.outdir, flags.src)
    tgts_file = load_bow.makeFile(flags.outdir, flags.tgt)
    x_npz_file = load_bow.makeFile(flags.outdir, flags.x_npz)
    dataset_file = load_bow.makeFile(flags.outdir, flags.dataset)

    km_labels = np.load(labels_file)
    X, X_indices = getXAndDataset('dev', x_npz_file, dataset_file)

    if flags.repro:
        centroids = reproClusters(km_labels, centroids_file)
    else:
        centroids = loadCentroids(centroids_file)

    eval(centroids, srcs_file, tgts_file, km_labels, X, X_indices)



if __name__ == '__main__':
    main()
