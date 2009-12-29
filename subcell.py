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
import skernel
import time
import getopt
import random
import classifier
import classman

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
        

def init_str_kernel(ds_names, ds_dir, k):
    krns = []
    i = 0
    for d in ds_names:
        ds = open(ds_dir + d + '.trn', 'r')
        krns.append(skernel.StrKernel(ds, k, d))
        ds.close()
        print len(krns[i].kgr),'k-grams in dataset', krns[i].lab
        i += 1
    return krns


def test_repr(c, ds):
    f = open('.tmp/' + c.lab + ds, 'r')
    rep = []
    for l in f:
        if l[0] != '>' and l[0] != '\n':
            rep.append(c.to_vector(l))
    return rep


def init_classifier(krns, ds_n):
    # SVM one-vs-all
    svmova = []
    startt = time.time()
    for k in krns:
        labels = []
        trn_ds = test_repr(k, '.trn')
        val_ds = test_repr(k, '.val')
        end_positive = len(trn_ds)
        for i in krns:
            if i!=k:
                trn_ds = trn_ds + test_repr(i, '.trn')
                val_ds = val_ds + test_repr(i, '.val')
        for j in xrange(len(trn_ds)):
            if j <= end_positive:
                labels.append(1)
            else:
                labels.append(0)
        # A questo punto abbiamo una serie di labels che indicizzano gli
        # esempi per un certo ds k.
        svm_k = classifier.Classifier(labels,trn_ds,val_ds)
        print "Classifier for %s initialized" % (k.lab)
        svmova.append(svm_k)
    print "Classifier initialized in %d seconds" % (time.time() - startt)
    print "There are %d samples for training and %d samples for validation" % (len(trn_ds), len(val_ds))
    return svmova 


def train(svmova):
    startt = time.time()
    for svm in svmova:
        svm.train()
    print "SVM trained in %d seconds" % (time.time() - startt)


def main():
    conf = read_opts(sys.argv)
    ds_n = os.listdir(conf['ds_dir'])
    split_dataset(conf['ds_dir'], ds_n, conf['t'], conf['v'])
    print 'Classifiers initialization:'
    krns = init_str_kernel(ds_n, '.tmp/', conf['k'])
    print '\nTesting vectorial representation:'
    for c in krns:
        str = time.time()
        test_repr(c,'.tst')
        print 'Test done in', time.time() - str, c.lab
    # Init SVM one-vs-all approach
    clm = classman.ClassMan(krns, ds_n)
    svm = clm.init_classifier()
    # Train SVM
    clm.train(svm, mt = True)
    # Prepare a sample
    sample = test_repr(krns[0],'.tst')[42]
    # Classify
    startt = time.time()
    result = svm[0].classify(sample)
    print "Sample",sample,"Prediction", result
    print "Classified in ", time.time() - startt,"s"
    #print "Sample test", sample
    print "Dimension of the sample", len(sample)
    clean_tmp()
    os.removedirs('.tmp')


if __name__ == "__main__":
    main()
