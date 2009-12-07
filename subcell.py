#!/usr/bin/python
'''
SubCell - Subcellular Proteine Localization learning algorithm

Usage:
    ./subcell.py <dataset_dir> [OPTIONS]

Options:
    -h  prints this help
    -t  training dataset percentage (float value)
    -v  validation dataset percentage (float value)
    -k  k-gram dimension list (ex. -k 2,3)
'''
import sys
import os
import classifier
import time
import getopt
import random

def read_opts(argv):
    res = {}
    res['t'] = 0.6
    res['v'] = 0.2
    res['k'] = [3]
    try:
        opts, args = getopt.gnu_getopt(argv, 't:v:k:h')
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, v in opts:
        if o == '-h':
            print __doc__
            sys.exit(0)
        elif o == '-t':
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
        print __doc__
        sys.exit(2)
    if res['ds_dir'][-1] != '/':
        res['ds_dir'] += '/'
    return res   

def clean_tmp():
    old = os.listdir('.tmp')
    for o in old:
        os.remove('.tmp/' + o)


def split_dataset(ds_dir, ds_names, t, v):
    r = random.Random()
        ds.close()
    try:
        os.mkdir('.tmp')
    except OSError:
        clean_tmp()
    for n in ds_names:
        src = open(ds_dir + n, 'r')
        dst_trn = open('.tmp/' + n + '.trn', 'w')
        dst_val = open('.tmp/' + n + '.val', 'w')
        dst_tst = open('.tmp/' + n + '.tst', 'w')
        tmp = src.readline()
        while tmp != '':
            if tmp != '\n':
                rnd = r.uniform(0,1)
                if rnd < t:
                    dst = dst_trn
                elif rnd < t + v:
                    dst = dst_val
                else:
                    dst = dst_tst
                dst.write(tmp)
                tmp = src.readline()
                dst.write(tmp)
            tmp = src.readline()
        dst_trn.close()
        dst_val.close()
        dst_tst.close()
        src.close()

    

        
def init_classifier(ds_names, ds_dir, k):
    cls = []
    i = 0
    for d in ds_names:
        ds = open(ds_dir + d + '.trn', 'r')
        cls.append(classifier.Classifier(ds, k, d))
        ds.close()
        print len(cls[i].kgr),'k-grams in dataset', cls[i].lab
        i += 1
    return cls



def main():
    conf = read_opts(sys.argv)
    ds_n = os.listdir(conf['ds_dir'])
    split_dataset(conf['ds_dir'], ds_n, conf['t'], conf['v'])
    print 'Classifiers initialization:'
    cls = init_classifier(ds_n, '.tmp/', conf['k'])
    print '\nTesting vectorial representation:'
    for c in cls:
        str = time.time()
        test_repr(c,'.tst')
        print 'Test done in', time.time() - str, c.lab
    clean_tmp()
    os.removedirs('.tmp')

def test_repr(c, ds):
    f = open('.tmp/' + c.lab + ds, 'r')
    rep = []
    for l in f:
        if l[0] != '>' and l[0] != '\n':
            rep.append(c.to_vector(l))

if __name__ == "__main__":
    main()
