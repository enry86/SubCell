# -*- coding: utf-8 -*-
''' Tuner functions '''

import os
import math

class Tuner:
    def __init__(self, classifier):
        self.classifier = classifier
	self.tuning_log = None

    def stochastic_local_search(self):
        None

    def iterative_tuner(self, start, end, step):
        '''
            Simple function to perform the validation in two passages, the
            first has a wide range and a big step, the second has a minor
            range and a little step
        '''
        data = []
        best = 0.
        C = start[0]
        while C <= end[0]:
            gamma = start[1]
            while gamma <= end[1]:
                #self.classifier.parameters.C = C
                #self.classifier.parameters.gamma = gamma
                self.classifier.update_parameters(C,gamma)
                self.classifier.train()
                line = "*** TUNING: C = %f; gamma = %f \n" % (C, gamma)
                self.log(line)
                c,w,nr,t = self.classifier.validate()
                try:
                    precision = float(c)/(c+w)
                except:
                    precision = 1.0
                recall = float(c)/(c+nr)
                
                print "Precision %f, Recall %f" % (precision, recall)
                if precision > best:
                    best = precision
                    data = [C, gamma]
                line = "Correct: %i / %i    C: %f    Gamma: %f \n" % (c, t, C, gamma)
                self.log(line)
                gamma += step[1]
            C += step[0]
        return (data, best)
        
    
    def parameter_search(self, element, mode):
        '''
            Returns the range and the step for the research mode
            If mode == 1, the is a finer search and check the new range is in
            the admitted
        '''
        if mode == 0:
            start = [self.classifier.C_range[0], self.classifier.gamma_range[0]]
            end = [self.classifier.C_range[1], self.classifier.gamma_range[1]]
            step = [self.classifier.C_step, self.classifier.gamma_step]
        else:
            c_start = element[0] - self.classifier.finer_range['C']
            c_end = element[0] + self.classifier.finer_range['C']

            gamma_start = element[1] - self.classifier.finer_range['gamma']
            gamma_end = element[1] + self.classifier.finer_range['gamma']

            if c_start < self.classifier.C_range[0]:
                c_start = self.classifier.C_range[0]
            if c_end > self.classifier.C_range[1]:
                c_end = self.classifier.C_range[1]
            if gamma_start < self.classifier.gamma_range[0]:
                gamma_start = self.classifier.gamma_range[0]
            if gamma_end > self.classifier.gamma_range[1]:
                gamma_end = self.classifier.gamma_range[1]

            start = [c_start, gamma_start]
            end = [c_end, gamma_end]
        
        C_step = float(end[0] - start[0])/self.classifier.n_iterations
        gamma_step = float(end[1] - start[1])/self.classifier.n_iterations
        step = [C_step, gamma_step]
        return (start, end, step)


    def log(self, data):
        '''
            Define the log function to store data in a file, default is the
            "tuning<class>.log"
            In case data is None, then the log will be close.
        '''
        try:
            os.mkdir("log")
        except OSError:
            pass

        if self.tuning_log == None:
            self.tuning_log = open("log/tuning-" + self.classifier.clabel + ".log",'w')

        if data == None:
            self.tuning_log.close()
        else:
            print "Write data ", data
            self.tuning_log.write(data)

    def tune(self, parameter):
        '''
            Considering default kernel as RBF - for the moment is the
            simplier.
            parameter:  if 0, then no parameters will be used;
                dicts of parameters:
                    Kernel  "stands for the Kernel type to use"
                    C       "stands for the C regulation parameter" 
                    gamma   "stands for the gamma kernel parameter"
            IMPROVEMENT: inserire parametri da riga di comando, filtrarli qui
            nei parametri
        '''
        #kernels = [LINEAR, POLY, RBF, SIGMOID]
        
        start, end, step = self.parameter_search(None, 0)
        line = ("Validation on %s \nCoarse: C = [%f, %f], step %f, gamma = " + \
               "[%f,  %f], step %f \n")  % (self.classifier.clabel, float(start[0]),
               float(end[0]), float(step[0]),\
                float(start[1]), float(end[1]), float(step[1]))
        self.log(line)
        best_param, prec = self.iterative_tuner(start, end, step)
        line = ("\nBest parameters found: C = %f, gamma = %f; The precision is " + \
               "about %f \n") % (float(best_param[0]), float(best_param[1]), prec)
        self.log(line)

        
        # Start a finer search on neighbour of the best parameter.
        start, end, step = self.parameter_search(best_param, 1)
        line = ("\n\n Validation on the finer range: C = [%f, %f], step %f, gamma = " + \
               "[%f, %f], step %f \n") % (float(start[0]), float(end[0]), float(step[0]), \
                float(start[1]), float(end[1]), float(step[1]))
        self.log(line)
        best_param_finer, prec_finer = self.iterative_tuner(start, end, step)
        
        # In the case the finer range founds the best parameter, it is swapped
        if prec_finer > prec:
            best_param = best_param_finer[:]
        
        line = ("\n\n\nBest parameters found: C = %f, gamma = %f; The precision is " + \
               "about %f \n") % (best_param[0], best_param[1], prec)
        self.log(line)
        line  = "Setting up the model with the parameters: C = %f, gamma = %f"  \
                % (best_param[0], best_param[1])
        self.log(line)
        self.classifier.update_parameters(best_param[0], best_param[1])
        self.classifier.train()
        self.log(None)
        print line
