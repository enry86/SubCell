#!/usr/bin/python

import sys
import os
import classifier
import time
import getopt

def read_opts(argv):
    res = {}
    res['t'] = 0.6
    res['v'] = 0.2
    res['k'] = [3]
    try:
        opts, args = getopt.gnu_getopt(argv, 't:v:k:')
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, v in opts:
        if o == '-t':
            try:
                res['t'] = float(v)
            except ValueError:
                print 'WARN: Value', v, 'invalid as split percentage'
        elif o == '-v':
            try:
                res['v'] = float(v)
            except ValueError:
                print 'WARN: Value', v, 'invalid as split percentage'
        elif o == '-k':
            res['k'] = v.split(',')
            try:
                for k in range(len(res['k'])):
                    res['k'][k] = int(res['k'][k])
            except ValueError:
                print 'WARN: Invalid k-gram dimension, falling to default'
                res['k'] = [3]
    if res['v'] + res['t'] >= 1:
        print 'WARN: No test dataset, falling back to default'
    try:
        res['ds_dir'] = args[1]
    except IndexError:
        print 'ERR: No dataset provided'
        sys.exit(2)
    if res['ds_dir'][-1] != '/':
        res['ds_dir'] += '/'
    return res   



        
def init_classifier(ds_names, ds_dir, k):
    cls = []
    i = 0
    for d in ds_names:
        cls.append(classifier.Classifier(open(ds_dir + d, 'r'), k, d))
        print len(cls[i].kgr), cls[i].lab
        i += 1
    return cls



def main():
    conf = read_opts(sys.argv)
    ds_n = os.listdir(conf['ds_dir'])
    cls = init_classifier(ds_n, conf['ds_dir'], conf['k'])
    print 'Testing vectorial representation:'
    str = time.time()
    #test_repr(cls[0])
    print 'Test done in', time.time() - str

def test_repr(c):
    f = open(sys.argv[1] + c.lab, 'r')
    rep = []
    for l in f:
        if l[0] != '>' and l[0] != '\n':
            rep.append(c.to_vector(l))

if __name__ == "__main__":
    main()
