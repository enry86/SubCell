#!/usr/bin/python

'''
Plot - Tool to represent data obtained by SubCell graphically

Usage:
    ./plot.py -d <directory1>[,<directories>,...]

Boundaries:
    directory1 must have a subdirectory /log/, where log file should be stored
'''


import sys
import os
import getopt
import string
import operator
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as npy
import matplotlib.pyplot as pyp


def read_opts(clp):
    '''
        Reads and returns command line parameters splitted.
    '''
    p = {}
    p['dir'] = []

    try:
        opts, args = getopt.gnu_getopt(clp, 'd:')
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for option, value in opts:
        if option == '-d':
            p['dir'] = value.split(',')
    return p

def initialization(clp):
    dlog = {}
    # Read the log directory
    for d in clp['dir']:
        logs = os.listdir(d + '/log')
        dlog[d] = logs

    # Build the plot directory
    try:
        os.mkdir('plot')
    except OSError:
        pass
    try:
        for d in dlog:
            os.mkdir('plot/' + d)
    except OSError:
        pass
    return dlog

def split_line(line):
    tmp = string.split(line,' ')
    for i in xrange(len(tmp)):
        if tmp[i] == 'C':
            C = float(tmp[i+2][:-1])
        elif tmp[i] == 'gamma':
            gamma = float(tmp[i+2][:-1])
        elif tmp[i] == 'F-Measure':
            f_meas = float(tmp[i+2])
    return C, gamma, f_meas

def read_data(dlog):
    '''
        Load log data in the memory.
        {'dir' : {'log file': [[C,gamma,F-measure],...,[...]]}}
    '''
    i = 0
    directories = {}
    for directory in dlog:
        log_data = {}
        for log in dlog[directory]:
            file_log = open(directory + '/log/' + log, 'r')
            values = []
            line = file_log.readline()
            i = 1
            while line != '':
                if line != '\n':
                    if line[0] == "*":
                        C,gamma,f_meas = split_line(line)
                        values.append([C, gamma, f_meas])
                line = file_log.readline()
                i += 1
            log_data[log] = values[:] 
        directories[directory] = log_data.copy()
    return directories


def plot(data):

    for d in data:
        for log in data[d]:
            pyp.figure(1)
            pyp.suptitle(d + ' - ' + log[:-4] + ' -- Gamma fixed < 0.20')
            C_dep = []
            gamma_dep = []
          
            for element in data[d][log]:
                if element[1] < 0.2:
                    C_dep.append([element[0], element[2]])
                 #C_dep.append([element[0], element[2]])
                 #gamma_dep.append([element[1], element[2]])

            C_dep.sort()
            #gamma_dep.sort()
            #print "GAMMA", gamma_dep
            C = map(operator.itemgetter(0), C_dep)
            #gamma = map(operator.itemgetter(0), gamma_dep)

            filename = (d + '-' + log[:-4])
            #pyp.subplot(211)
            pyp.ylabel('F-measure')
            pyp.xlabel('C')
            pyp.plot(C, map(operator.itemgetter(1), C_dep))
            #print "FMEASURE", map(operator.itemgetter(1), C_dep)

            #pyp.subplot(212)
            #pyp.ylabel('F-measure')
            #pyp.xlabel('gamma')
            #pyp.plot(gamma, map(operator.itemgetter(1), gamma_dep))
            #print "FMEASURE2", map(operator.itemgetter(1), gamma_dep)           

            pyp.savefig(filename)
            pyp.clf()

def main():
    clp = read_opts(sys.argv)
    dlog = initialization(clp)
    data = read_data(dlog)
    plot(data)

if __name__ == '__main__':
    main()
