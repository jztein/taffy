import json

def loadData(infile):
    with open(infile, 'r') as f:
        indata = json.load(f)
    data = []
    for thread in indata.itervalues():
        for msg in thread['thread']:
            data.append(msg['msg'])
    return data
