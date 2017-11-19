"""Parses data to standard.

Example usage:
$ python data_parser.py smsexport ../data/sms.xml ../out/

Each sms message is represented by:
thread_id (other person(s)' phone number)
thread_name (other person(s)' contact names)
date (send date)
msg (SMS message)
"""
import argparse
import json
import os
import pickle

import numpy as np
import xml.etree.ElementTree as ET

from sklearn.feature_extraction.text import TfidfVectorizer


flag_parser = argparse.ArgumentParser()
flag_parser.add_argument('cmd')
flag_parser.add_argument('infile')
flag_parser.add_argument('outdir')
flags = flag_parser.parse_args()

class SmsExportParser(object):

    def __init__(self):
        self.data = {}  # Use same object for perf.
        self.outsuffix = '_se.json'
        self.numMsgs = 0

    def convertFile(self, infile, outfile):
        tree = ET.parse(infile)
        root = tree.getroot()
        for mms in root.iter('mms'):
            self.parseElement(mms, is_sms=False)
            self.numMsgs += 1
        for sms in root.iter('sms'):
            self.parseElement(sms)
            self.numMsgs += 1
        outfile += self.outsuffix
        with open(outfile, 'w') as f:
            json.dump(self.data, f, indent=4, separators=(',', ':'))
        print('Wrote to %s, threads: %s, msgs: %s' % (
                outfile, len(self.data), self.numMsgs))

    def parseElement(self, sms, is_sms=True):
        thread_id = sms.get('address').strip('+ ')
        thread_id = thread_id.replace(' ', '')
        thread_id = thread_id.replace('(', '')
        thread_id = thread_id.replace(')', '')
        thread_id = thread_id.replace('-', '')

        thread_name = sms.get('contact_name')
        date = sms.get('date')

        if is_sms:
            msg = sms.get('body')
        else:  # Is MMS. Message is in parts.part[1].
            parts = sms.find('parts')
            if not parts:
                return
            msg = []
            for part in parts.findall('part'):
                if part.get('ct') == 'text/plain':
                    text = part.get('text')
                    if not text: continue
                    msg.append(text)
            if not msg:
                return
            msg = '.'.join(msg)

        # Add to json
        if thread_id not in self.data:
            self.data[thread_id] = {'name': thread_name, 'thread': []}
        self.data[thread_id]['thread'].append({'msg': msg, 'date': date})


def loadData(infile):
    with open(infile, 'r') as f:
        indata = json.load(f)
    data = []
    for thread in indata.itervalues():
        for msg in thread['thread']:
            data.append(msg['msg'])
    return data


def preprocess(s):  # Can't use lambda or else can't pickle non-main function.
    return s.lower()


class BOWParser(object):
    """Bag of Words parser."""

    def __init__(self):
        self.outsuffix = '_bow.npy'
        self.vectorizer_suffix = self.outsuffix + '_vec.pk'

    def convertFile(self, infile, outfile):
        data = loadData(infile)
        vectorizer = TfidfVectorizer(preprocessor=preprocess)
        X = vectorizer.fit_transform(data)
        X_file = outfile + self.outsuffix
        with open(X_file, 'w') as f:
            np.save(f, X)
        print('Wrote to %s, TF-idf samples: %s, features: %s' % (
                X_file, X.shape[0], X.shape[1]))
        pickle.dump(vectorizer, open(outfile + self.vectorizer_suffix, 'wb'))
        

def toOutfile(infile, outdir):
    name = infile.split('/')[-1]
    return os.path.join(outdir, name)

def main():
    if flags.cmd == 'smsexport':
        parser = SmsExportParser()
    elif flags.cmd == 'bow':
        parser = BOWParser()
    else:
        print 'Provide cmd lol'
        return

    outfile = toOutfile(flags.infile, flags.outdir)
    parser.convertFile(flags.infile, outfile)


if __name__ == '__main__':
    main()
