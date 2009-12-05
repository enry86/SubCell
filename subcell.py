#!/usr/bin/python

import sys
import os
import classifier
import time

def main():
    k = 3
    ds_dir = sys.argv[1]
    ds_names = os.listdir(ds_dir)
    cls = []
    i = 0
    for d in ds_names:
        cls.append(classifier.Classifier(open(ds_dir + d, 'r'), k, d))
        print len(cls[i].kgr), cls[i].lab
        i += 1
    print 'Testing vectorial representation:'
    str = time.time()
    test_repr(cls[0])
    print 'Test done in', time.time() - str

def test_repr(c):
    f = open(sys.argv[1] + c.lab, 'r')
    rep = []
    for l in f:
        if l[0] != '>' and l[0] != '\n':
            rep.append(c.to_vector2(l))

if __name__ == "__main__":
    main()
