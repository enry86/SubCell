#!/usr/bin/python

'''
SubCell - Subcellular Protein Localization Learning Algorithm

Usage:
    ./subcell.py <model_dir> [OPTIONS]

Options:
    -h prints this help
    -f input file
    -o output file
'''

import skernel
import os
import getopt
import sys
from svm import *

class Model:
    def __init__(self, filename):
        self.model = svm_model(filename)

    def classify(self, name, vec):
        pred = self.model.predict_probability(vec)
        return pred


def read_opts(argv):
    '''
        Command-Line options interpretation
    '''
    res = {}
    res['out_file'] = False 
    res['in_file'] = False
    try:
        opts, args = getopt.gnu_getopt(argv, 'f:o:h')
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, v in opts:
        if o == '-h':
            print __doc__
            sys.exit(0)
        elif o == '-o':
            res['out_file'] = True
            res['o'] = v
        elif o == '-f':
            res['in_file'] = True
            res['f'] = v
    try:
        res['m'] = args[1]
    except IndexError:
        print 'ERR: No model provided'
        print __doc__
        sys.exit(2)
    return res


def init_classifier(m_dir):
    res = {}
    for m in os.listdir(m_dir):
        res[m] = Model(m_dir + '/' + m)
    return res


def init_kernels(m_dir):
    res = {}
    for k in os.listdir(m_dir):
        f = open(m_dir + '/' + k, 'r')
        tmp[k] = pickle.load(f)
        f.close()
    return res

def main():
    conf = read_opts(sys.argv)
    svms = init_classifier(conf['m'])
    krns = init_kernels(conf['m'])



if __name__ == '__main__':
    main()
