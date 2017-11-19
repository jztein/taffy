"""Do kmeans on BOW vecs."""
import argparse
import json
import pickle
from time import time
import load_bow

from numpy import __version__ as nv
print ('numpy version:', nv)
from sklearn import __version__ as skv
print ('sklearn version:', skv)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

COLORS = ['bgrcmk']

flag_parser = argparse.ArgumentParser()
flag_parser.add_argument('n')
flag_parser.add_argument('bowfile')
flag_parser.add_argument('vectorizer')
flags = flag_parser.parse_args()

def loadData(infile):
    return load_bow(infile)


def getBOW(bowfile):
    data = loadData(bowfile)
    vectorizer = CountVectorizer(preprocessor=lambda s: s.lower())
    #vectorizer = TfidfVectorizer(preprocessor=lambda s: s.lower())
    X = vectorizer.fit_transform(data)
    print('TF-idf samples: %s, features: %s' % X.shape)

    features = vectorizer.get_feature_names()
    pickle.dump(features, open('../out/bow_features.pk', 'wb'))
    return X, vectorizer

def main():
    print('Using bowfile %s' % flags.bowfile)
    X, vectorizer = getBOW(flags.bowfile)

    n = int(flags.n)
    km = KMeans(n_clusters=n, init='k-means++', n_init=10)

    t0 = time()
    km.fit(X)
    print('Kmeans took %.3fs' % (time() - t0))

    np.save('../out/X_labels.npy', km.labels_)

    printStats(km, vectorizer)  # Top features

    t0_t = time()
    embeddings = TSNE(n_components=2)#tsne_num_components)
    Y = embeddings.fit_transform(X.todense())
    np.save('../out/tsne.npy', Y)
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
