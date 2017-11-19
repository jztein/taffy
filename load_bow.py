import json

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
            data.append({'x': msg['msg'], 'y': nextMsg['msg']})
    return data
