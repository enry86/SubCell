#!/usr/bin/python

'''
SubCell - Subcellular Protein Localization Learning Algorithm

Usage:
    ./subcell.py <model_dir> <input_file> [OPTIONS]

Options:
    -h prints this help
    -o output file
'''

import skernel
import pickle
import os
import getopt
import sys
from svm import *

class Model:
    def __init__(self, filename):
        self.model = svm_model(filename)

    def classify(self, vec):
        pred = self.model.predict_probability(vec)
        return pred


def read_opts(argv):
    '''
        Command-Line options interpretation
    '''
    res = {}
    res['out_file'] = None
    try:
        opts, args = getopt.gnu_getopt(argv, 'o:h')
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, v in opts:
        if o == '-h':
            print __doc__
            sys.exit(0)
        elif o == '-o':
            res['out_file'] = v
    try:
        res['m'] = args[1]
    except IndexError:
        print 'ERR: No model provided'
        print __doc__
        sys.exit(2)
    try:
        res['in_file'] = args[2]
    except IndexError:
        print 'ERR: No input provided'
        print __doc__
        sys.exit(2)
    return res


def init_classifier(m_dir):
    res = {}
    for m in os.listdir(m_dir):
        if m.split('.')[1] == 'mdl':
            res[m.split('.')[0]] = Model(m_dir + '/' + m)
    return res


def classify(svms, vec):
    best = 0.0
    res = None
    for s in svms:
        tmp = svms[s].classify(vec)
        if tmp[1][1] > best:
            best = tmp[1][1]
            res = s
    return res, best


def init_kernel(m):
    f = open(m + '/' + m + '.krn', 'r')
    ker = pickle.load(f)
    f.close()
    return ker


def output(name, cl, pr, o):
    out = name + '\t' + cl + '\t' + str(pr)
    if o == None:
        print out
    else:
        o.write(out + '\n')


def start_class(svms, krn, in_f, out_f):
    f = open(in_f, 'r')
    if out_f != None:
        o = open(out_f, 'w')
    else:
        o = None
    l = f.readline()
    while l != '':
        name = l.strip()
        prot = f.readline().strip()
        cl, pr = classify(svms, krn.to_vector(prot))
        output(name, cl, pr, o)
        l = f.readline()
    f.close()
    if o != None:
        o.close()


def main():
    conf = read_opts(sys.argv)
    svms = init_classifier(conf['m'])
    krn = init_kernel(conf['m'])
    start_class(svms, krn, conf['in_file'], conf['out_file'])


if __name__ == '__main__':
    main()
