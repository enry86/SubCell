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
    -C  fixed C parameter for svm training
    -g  gamma parameter for the RBF kernel
    -n  number of iterations to validate the dataset in the optimal
    paramters search
    -m  alternative filename for model
'''

import sys
import os
import skernel
import time
import getopt
import random
import classman
import pickle

def read_opts(argv):
    '''
        commandline options parsing
    '''
    res = {}
    res['t'] = 0.6
    res['v'] = 0.2
    res['k'] = [3]
    res['m'] = 'model'
    res['C'] = 1
    res['g'] = 0.1
    res['n'] = 1
    try:
        opts, args = getopt.gnu_getopt(argv, 'm:t:v:k:h')
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, v in opts:
        if o == '-h':
            print __doc__
            sys.exit(0)
        elif o == '-m':
            res['m'] = v
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
    '''
        removes the temporary files of the split dataset
    '''
    old = os.listdir('.tmp')
    for o in old:
        os.remove('.tmp/' + o)


def split_dataset(ds_dir, ds_names, t, v):
    '''
        splits the dataset according to the configuration
    '''
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
    '''
        String kernel initialization
    '''
    krns = []
    i = 0
    for d in ds_names:
        ds = open(ds_dir + d + '.trn', 'r')
        krns.append(skernel.StrKernel(ds, k, d))
        ds.close()
        print '\t', len(krns[i].kgr),'k-grams in dataset', krns[i].lab
        i += 1
    return krns


def to_disk(svms, krn, filename):
    '''
        Stores classifier to disk
    '''
    try:
        os.mkdir(filename)
    except OSError:
        pass
    for s in svms:
        s.model.save(filename + '/' + s.clabel + '.mdl')
    f = open(filename + '/' + filename + '.krn', 'w')
    pickle.dump(krn,f)
    f.close()


def output_metrics(met):
    print 'Quality measures for each SVM:'
    for m in met[0]:
        print '\t' + m + ':'
        print '\t\tPrecision:', met[0][m][0]
        print '\t\tRecall:', met[0][m][1]
    print '\nMicro-average:'
    print '\tPrecision:', met[1][0]
    print '\tRecall:', met[1][1]


def main():
    conf = read_opts(sys.argv)
    ds_n = os.listdir(conf['ds_dir'])
    split_dataset(conf['ds_dir'], ds_n, conf['t'], conf['v'])
    print 'Kernel initialization:'
    krn = skernel.StrKernel(ds_n, '.tmp/', conf['k'])
    print '\tString kernel initialized with %s k-grams' % len(krn.kgr)
    # Init SVM one-vs-all approach
    clm = classman.ClassMan(krn, ds_n)
    clm.init_classifier()
    # Train SVM
    clm.train(mt = True)
    #clm.train(mt = False)
    # perform test
    clm.test()
    clm.validation(0)
    clm.test()
    output_metrics(clm.get_metrics())
    to_disk(clm.svms, krn, conf['m'])
    print 'Model saved with filename:', conf['m']
    clean_tmp()
    os.removedirs('.tmp')


if __name__ == "__main__":
    main()
