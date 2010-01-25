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
                self.classifier.update_parameters(C,gamma, None)
                self.classifier.train()
                line = "*** TUNING: C = %f; gamma = %f " % (C, gamma)
                self.log(line)
                c,w,nr,t = self.classifier.validate()
                try:
                    precision = float(c)/(c+w)
                except:
                    precision = 1.0
                recall = float(c)/(c+nr)
                try:
                    f_meas = 2.0 * (recall * precision) / (recall+precision)
                except:
                    f_meas = 0.0
                print "Precision %f, Recall %f" % (precision, recall)
                if f_meas > best:
                    best = f_meas
                    data = [C, gamma]
                line = "Correct: %i / %i    C: %f    Gamma: %f \n" % (c, t, C, gamma)
                self.log(line)
                gamma += step[1]
            C += step[0]
        
        # FIX in the case the classifier fail for whole validation
        if data == []:
            mid_C = (self.C_range[1] + self.C_range[0])/2
            mid_gamma = (self.gamma_range[1] + self.gamma_range[0])/2
            data = [mid_C, mid_gamma]
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
        # 1 = Polynomial; 2 = Radial Basis Function; 3 = Sigmoid 
        kernels = [1, 2, 3]
        kernels_n = ['Polynomial', 'RBF', 'Sigmoid']
        performance = [None, None, None]

        line = ("Validation on %s") % (self.classifier.clabel)
        self.log(line)

        try:
            for kernel in kernels:
	        self.classifier.parameters.kernel_type = kernel
		
		start, end, step = self.parameter_search(None, 0)
		print "PARAM:", start, end, step
		line = ("Kernel %s \nCoarse: C = [%f, %f], step %f, gamma = " + \
		  "[%f,  %f], step %f \n")  % (kernels_n[kernel - 1], float(start[0]),
		  float(end[0]), float(step[0]),\
		    float(start[1]), float(end[1]), float(step[1]))
		self.log(line)
		best_param, prec = self.iterative_tuner(start, end, step)
		print "BEST", best_param, prec
		line = ("\nBest parameters found: C = %f, gamma = %f; F-Measure is " + \
		  "about %f \n") % (float(best_param[0]), float(best_param[1]), prec)
		self.log(line)


		# Start a finer search on neighbour of the best parameter.
		start, end, step = self.parameter_search(best_param, 1)
		print "PARAM finer",start,end,step
		line = ("\n\n Validation on the finer range: C = [%f, %f], step %f, gamma = " + \
		  "[%f, %f], step %f \n") % (float(start[0]), float(end[0]), float(step[0]), \
		    float(start[1]), float(end[1]), float(step[1]))
		self.log(line)
		best_param_finer, prec_finer = self.iterative_tuner(start, end, step)

		# In the case the finer range founds the best parameter, it is swapped
		if prec_finer > prec:
		    best_param = best_param_finer

		print "BEST ALL", best_param, prec
		line = ("\n\n\nBest parameters found: C = %f, gamma = %f; F-measure is " + \
		  "about %f \n") % (best_param[0], best_param[1], prec)
		self.log(line)
		performance[kernel - 1] = [prec, best_param]

        except e:
            print "EXCEPTION", e
	    print "KERNEL", kernel
	    print "START", start
	    print "END", end
	    print "STEP", step
	    print "PRECISION", prec
	    print "COARSE PARAMETERS", best_param
	    print "FINER PRECISION %s PARAMETER %s" %(prec_finer, best_param_finer)
	    print "KERNELS PERFORMANCE", performance
        
        best = 0
        for i in xrange(len(kernels)):
            if performance[i][0] > performance[best][0]:
                best = i
#
#	for k in kernels:
#	    if performance[k] != None:
#	        if performance[k][0] > performance[best][0]:
#		    best = k

	line  = ("Setting up the model build on the kernel %s with the parameters" + \
		": C = %f, gamma = %f; F-measure is about %f") \
		% (kernels_n[best],performance[best][1][0], performance[best][1][1], \
			performance[best][0])
	print line
	self.log(line)
	self.classifier.update_parameters(performance[best][1][0],performance[best][1][1],best)
	self.classifier.train()
	print "MODEL TRAINED"
	self.log(None)
