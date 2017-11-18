"""Parses data to standard.

Each sms message is represented by:
thread_id (other person(s)' phone number)
thread_name (other person(s)' contact names)
date (send date)
msg (SMS message)
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

flag_parser = argparse.ArgumentParser()
flag_parser.add_argument('infile')
flag_parser.add_argument('outdir')
flags = flag_parser.parse_args()

SMIL_START = '<smil>'
SMIL_END = '</smil>'
SMIL_END_LEN = len(SMIL_END)

class SmsExportParser(object):

    def __init__(self):
        self.data = {}  # Use same object for perf.

    def convertFile(self, infile, outfile):
        tree = ET.parse(infile)
        root = tree.getroot()
        for mms in root.iter('mms'):
            self.parseElement(mms, is_sms=False)
        for sms in root.iter('sms'):
            self.parseElement(sms)
        with open(outfile, 'w') as f:
            json.dump(self.data, f, indent=4, separators=(',', ':'))

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
                print 'Unexpected MMS parts', parts[0].attrib
                return
            msg = '.'.join(msg)

        # Add to json
        if thread_id not in self.data:
            self.data[thread_id] = {'name': thread_name, 'thread': []}
        self.data[thread_id]['thread'].append({'msg': msg, 'date': date})

def toOutfile(infile, outdir):
    name = infile.split('/')[-1]
    return os.path.join(outdir, name)

def main():
    parser = SmsExportParser()
    outfile = toOutfile(flags.infile, flags.outdir)    
    parser.convertFile(flags.infile, outfile)


if __name__ == '__main__':
    main()
