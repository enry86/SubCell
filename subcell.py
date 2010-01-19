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

import getopt
import sys


def read_opts(argv):
    '''
        Command-Line options interpretation
    '''
    res = {}
    res{'out_file'} = False 
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


def main():
    conf = read_opts(sys.argv)





if __name__ == '__main__':
    main()
