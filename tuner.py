# -*- coding: utf-8 -*-
''' Tuner functions '''

import os
import math

class Tuner:
    def __init__(self, classifier):
        self.classifier = classifier
	self.tuning_log = None


    def iterative_tuner(self, start, end, step):
        '''
            Perform the validation in two passages, the
            first has a wide range and a big step, the second has a minor
            range and a little step
        '''
        data = []
        best = 0.
        C = start[0]
        while C <= end[0]:
            gamma = start[1]
            while gamma <= end[1]:
                self.classifier.measure.reset()
                self.classifier.update_parameters(C,gamma)
                self.classifier.train()
                self.classifier.validate()
                precision,recall,f_meas = \
                    self.classifier.measure.svm_metrics(self.classifier.clabel)
                line = ("*** TUNING: C = %f; gamma = %f; Precision = %f; " +\
                        "Recall = %f; F-Measure = %f\n") \
                        % (float(C), gamma, precision, recall, f_meas)
                self.log(line)
                print "Precision %f, Recall %f, F-Measure %f" % \
                        (precision, recall, f_meas)
                if f_meas > best:
                    best = f_meas
                    data = [C, gamma]
                c,t = self.classifier.measure.all_counter()
                line = "Correct: %i / %i    C: %f    Gamma: %f \n" % (c, t, C, gamma)
                gamma += step[1]
            C += step[0]
        
        # FIX in the case the classifier fail for whole validation
        if data == []:
            mid_C = (self.classifier.C_range[1] + \
                self.classifier.C_range[0])/2
            mid_gamma = (self.classifier.gamma_range[1] + \
                self.classifier.gamma_range[0])/2
            data = [mid_C, mid_gamma]
        return (data, best)


    def parameter_search(self, element, mode):
        '''
            Returns the range and the step for the research mode
            If mode == 1, there is a finer search and check the new range is in
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
       
        # In the case the value is set, both value of the rage are the same,
        # so the step is the value to out the range
        if self.classifier.C_range[0] == self.classifier.C_range[1]:
            C_step = self.classifier.C_range[0]
        else:
            C_step = float(end[0] - start[0])/self.classifier.n_iterations

        if self.classifier.gamma_range[0] == self.classifier.gamma_range[1]:
            gamma_step = self.classifier.gamma_range[0]
        else:
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


    def tune(self):
        '''
            Considering default kernel as RBF. The tuning consist to find the
            best C, gamma parameter to maximize the F-measure for the SVM.
            #In particular if one of the two parameter is "None", then the
            #function iterate for its default range foreseen by the SVM.
            #If both parameters are already fixed, this tuning should never be
            #computed
        '''

        line = ("Validation on %s") % (self.classifier.clabel)
        self.log(line)

		
        start, end, step = self.parameter_search(None, 0)
        #print "PARAM:", start, end, step
        line = ("Coarse: C = [%f, %f], step %f, gamma = " + \
            "[%f,  %f], step %f \n")  \
            % (float(start[0]), float(end[0]), float(step[0]),\
            float(start[1]), float(end[1]), float(step[1]))
        self.log(line)
        best_param, prec = self.iterative_tuner(start, end, step)
        #print "BEST", best_param, prec
        line = ("\nBest parameters found: C = %f, gamma = %f; F-Measure is " + \
          "about %f \n") % (float(best_param[0]), float(best_param[1]), prec)
        self.log(line)
        start, end, step = self.parameter_search(best_param, 1)
    	# Start a finer search on neighbour of the best parameter.
	    #start, end, step = self.parameter_search(best_param, 1)
        # Execute the finer search if there is at least one parameter to
        # iterate
        if (step[0] != start[0]) and (step[1] != start[1]):
            #print "PARAM finer",start,end,step
            line = ("\n\n Validation on the finer range: C = [%f, %f], step %f, gamma = " + \
              "[%f, %f], step %f \n") % (float(start[0]), float(end[0]), float(step[0]), \
                float(start[1]), float(end[1]), float(step[1]))
            self.log(line)
            best_param_finer, prec_finer = self.iterative_tuner(start, end, step)

            # In the case the finer range founds the best parameter, it is swapped
            if prec_finer > prec:
                best_param = best_param_finer[:]
                prec = prec_finer

            print "BEST ALL", best_param, prec
            line = ("\n\n\nBest parameters found: C = %f, gamma = %f; F-measure is " + \
              "about %f \n") % (best_param[0], best_param[1], prec)
            self.log(line)

	line  = ("Setting up the model for %s with the parameters: " + \
                "C = %f, gamma = %f; F-measure is about %f") \
		% (self.classifier.clabel,best_param[0], best_param[1], prec)
	print line
	self.log(line)
	self.classifier.update_parameters(best_param[0],best_param[1])
	self.classifier.train()
	print "MODEL TRAINED"
	self.log(None)
