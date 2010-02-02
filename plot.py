#!/usr/bin/python

'''
Plot -  Tool to represent data obtained by SubCell graphically
        Standard behaviour is to plot the F-measure bounded to the C with
        gamma < 0.00020
Usage:
    ./plot.py -d <directory1>[,<directories>,...] [OPTIONS]

Boundaries:
    directory1 must have a subdirectory /log/, where log file should be stored
    cannot be both C and gamma fixed

Options
    -C      Set the fixed C parameter
    -g      Set the fixed gamma paramter
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
        clp     command line parameters
        Return
            p['dir']    list of logs directories
    '''
    p = {}
    p['dir'] = []
    p['C'] = None
    p['g'] = None
    try:
        opts, args = getopt.gnu_getopt(clp, 'd:C:g:')
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for option, value in opts:
        if option == '-d':
            p['dir'] = value.split(',')
        if option == '-C':
            try:
                if float(value) >= 0:
                    p['C'] = float(value)
            except ValueError:
                print "WARN: Invalid value for C parameter."
        if option == '-g':
            try:
                if float(value) >= 0:
                    p['g'] = float(value)
            except ValueError:
                print "WARN: Invalid value for gamma parameter."
    if ( p['C'] != None ) and ( p['g'] != None):
            print "Error: Both parameter fixed. Choose one."
            sys.exit(1)
    return p


def initialization(clp):
    '''
        Read logs directory and build directory plot with subdirs
        clp     command line parameter
    '''
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
    for d in dlog:
        try:
            os.mkdir('plot/' + d)
        except OSError, e:
            print e
            pass
    return dlog


def split_line(line):
    '''
        Given a log line as input, split it for the " " and returns the values
        Line format:
           *** TUNING ..... C = <float> ... gamma = <float> ... F-Measure = <float>
        Returns:
            C, gamma, F-Measure
    '''
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


def clean(values):
    '''
        Remove from array the duplicate elements for the C
    '''
    i = 0
    while 1:
        try:
            if abs(values[i][0] - values[i+1][0]) < 1000:
                if values[i][1] < values[i+1][1]:
                    values.__delitem__(i)
                else:
                    values.__delitem__(i+1)
            else:
                i += 1
        except IndexError:
            break


def plot(clp,data):
    '''
        Given the whole log data, plots the graph for each log directory,
        considering each SVM log separately. Gamma is fixed < 0.00020
        due the relevant results are in this area
            x Axis      C
            y Axis      F-measure
    '''
    for d in data:
        for log in data[d]:
            pyp.figure(1)
            values = []
            
            ind = 1
            threshold = 0.00020
            #txt = "Gamma fixed < " + str(threshold)
            if clp['C'] != None:
                ind = 0
                threshold = clp['C']
            elif clp['g'] != None:
                ind = 1
                threshold = clp['g']
            pyp.suptitle(log[7:-4])

            for element in data[d][log]:
                if element[ind] < threshold:
                    values.append([element[ (ind - 1) % 2 ], element[2]])

            values.sort()
            clean(values)
            x = map(operator.itemgetter(0), values)

            filename = ('plot/' + d + '/' + d + '-' + log[:-4])
            pyp.ylabel('F-measure')
            if clp['C'] != None:
                pyp.xlabel('gamma')
            else:
                pyp.xlabel('C')
            pyp.plot(x, map(operator.itemgetter(1), values))

            print "Saving: ", filename
            pyp.savefig(filename)
            pyp.clf()

def main():
    clp = read_opts(sys.argv)
    dlog = initialization(clp)
    data = read_data(dlog)
    plot(clp,data)

if __name__ == '__main__':
    main()
