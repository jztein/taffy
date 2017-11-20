import os
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def makeFile(dir, name):
    return os.path.join(dir, name)


def vectorize(data, tfidf=True):
    if tfidf:
        vectorizer = TfidfVectorizer(preprocessor=lambda s: s.lower())
    else:
        vectorizer = CountVectorizer(preprocessor=lambda s: s.lower())
    X = vectorizer.fit_transform(data)
    print('TF-idf samples: %s, features: %s' % X.shape)
    return X, vectorizer


def loadLines(infile):
    with open(infile, 'r') as f:
        return f.readlines()


def loadData(infile):
    with open(infile, 'r') as f:
        indata = json.load(f)
    data = []
    for thread in indata.itervalues():
        for msg in thread['thread']:
            data.append(msg['msg'])
    return data


def loadAndConvertVanillaTrain(infile):
    with open(infile, 'r') as f:
        indata = json.load(f)
    data = []
    TWENTY_FOUR_HOURS = 86400000  # In milliseconds.
    for thread in indata.itervalues():
        lastMsgIndex = len(thread['thread']) - 1
        for i, msg in enumerate(thread['thread']):
            if i == lastMsgIndex:
                continue
            nextMsg = thread['thread'][i+1]
            if int(nextMsg['date']) - int(msg['date']) > TWENTY_FOUR_HOURS:
                continue
            data.append({
                    'x': msg['msg'].encode('utf-8').lower(),
                    'y': nextMsg['msg'].encode('utf-8').lower()})
    return data



def loadVanillaTrainVocab(data):
    vocab = set()
    for xy in data:
        vocab.update([w.strip(' .,!?~') for w in xy['x'].split()])
        vocab.update([w.strip(' .,!?~') for w in xy['y'].split()])
    return list(vocab)
