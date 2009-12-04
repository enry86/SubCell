#!/usr/bin/python

import sys
import os
import classifier

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

if __name__ == "__main__":
    main()
