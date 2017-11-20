"""Do kmeans on BOW vecs."""
import argparse
import json
import pickle
import random
from time import time
import load_bow

from numpy import __version__ as nv
print ('numpy version:', nv)
from sklearn import __version__ as skv
print ('sklearn version:', skv)

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

COLORS = ['bgrcmk']

SUFFIX = '_train1'

flag_parser = argparse.ArgumentParser()
flag_parser.add_argument('n')
flag_parser.add_argument('bowfile')
flag_parser.add_argument('vectorizer')
flag_parser.add_argument('--get_X')
flag_parser.add_argument('--no_json')
flags = flag_parser.parse_args()

def loadData(infile):
    if flags.no_json:
        return load_bow.loadLines(infile)

    return load_bow.loadData(infile)


def getBOW(bowfile):
    data = loadData(bowfile)
    #vectorizer = CountVectorizer(preprocessor=lambda s: s.lower())
    vectorizer = TfidfVectorizer(preprocessor=lambda s: s.lower())
    X = vectorizer.fit_transform(data)
    print('TF-idf samples: %s, features: %s' % X.shape)

    features = vectorizer.get_feature_names()
    pickle.dump(features, open('../out/bow_features' + SUFFIX + '.pk', 'wb'))
    return X, vectorizer


dataset_file = '../out/kmeans_dataset' + SUFFIX + '.pk'

def makeDataset(num_data):
    indices = [i for i in xrange(num_data)]
    random.shuffle(indices)
    eighty_p = int(.8 * num_data)
    ninety_p = int(.9 * num_data)
    train, dev, test = [], [], []
    for i, index in enumerate(indices):
        if i < eighty_p:
            train.append(index)
        elif i < ninety_p:
            dev.append(index)
        else:
            test.append(index)

    dataset = (train, dev, test)
    pickle.dump(dataset, open(dataset_file, 'wb'))
    return dataset

def getTrainDataset():
    train, dev, test = pickle.load(open(dataset_file, 'rb'))
    return train

def main():
    print('Using bowfile %s' % flags.bowfile)
    X_file = '../out/X' + SUFFIX + '.npz'
    if flags.get_X:
        X, vectorizer = getBOW(flags.bowfile)
        save_npz(X_file, X)
        train_dataset, _, _ = makeDataset(X.shape[0])
    else:
        X = load_npz(X_file)
        vectorizer = pickle.load(open('../out/bow_features' + SUFFIX + '.pk',
                                      'rb'))
        train_dataset = getTrainDataset()
    # X to fit
    X = X[train_dataset]

    n = int(flags.n)
    km = KMeans(n_clusters=n, init='k-means++', n_init=10)

    t0 = time()
    km.fit(X)
    print('Kmeans took %.3fs' % (time() - t0))

    np.save('../out/X_labels' + SUFFIX + '.npy', km.labels_)
    label_map = {}
    for t_i, label in enumerate(km.labels_):
        if str(label) not in label_map:
            label_map[str(label)] = []
        label_map[str(label)].append(t_i)
    pickle.dump(label_map, open('../out/X_label_map' + SUFFIX + '.pk', 'wb'))
    np.save('../out/km_centroids' + SUFFIX + '.npy', km.cluster_centers_)

    if True:
        return

    printStats(km, vectorizer)  # Top features

    if False:
        t0_t = time()
        embeddings = TSNE(n_components=2)#tsne_num_components)
        Y = embeddings.fit_transform(X.todense())
        np.save('../out/tsne' + SUFFIX + '.npy', Y)
        for i in xrange(n):
            plt.scatter(Y[km.labels_==i, 0], Y[km.labels_==i, 1], c=COLORS[i])
        plt.show()
        print('TSNE took %.3fs' % (time() - t0_t))


def printStats(km, vectorizer):
    print 'CENTERS coords:', km.cluster_centers_
    for i, a in enumerate(km.cluster_centers_):
        for j, b in enumerate(km.cluster_centers_):
            if i <= j: continue
            print 'CENTERs ', i, ',', j, ' dist: ', np.linalg.norm(a-b)
    order_centroids = []
    for center in km.cluster_centers_:
        order_centroids.append(center.argsort()[::-1])
    print
    features = vectorizer.get_feature_names()
    seen = set()
    num_pts = 0
    n = len(order_centroids) # num clusters
    find_top_iters = {centroid_label: 0 for centroid_label in xrange(n)}
    for centroid_label, cluster in enumerate(order_centroids):
        num_pts += len(cluster)
        print 'Cluster', cluster[:5]
        top = 0
        for i, feature_index in enumerate(cluster):
            find_top_iters[centroid_label] += 1
            if i > 100:
                print 'Exceeded 100 before finding top for centroid ', i
                break
            if centroid_label == km.labels_[feature_index]:
                if feature_index in seen:
                    print 'Seen before:', features[feature_index]
                    continue
                seen.add(feature_index)
                top += 1
                print('Top word: %s' % features[feature_index])
                if top == 5:
                    break
    print 'Top:', [features[i] for i in seen]
    print 'WHAT: ', num_pts, ' != ', len(features)
    print 'iters to find top:', find_top_iters

if __name__ == '__main__':
    main()
