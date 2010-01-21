# -*- coding: utf-8 -*-
'''Classifier functions'''

from svm import *
import math

class Classifier:
    def __init__(self, training_labels, training, validation_labels, validation, clabel):
        '''
            Initialization of the problem through training data.
            labels      is a list of 0,1 that recognize each sample in the
                        training, where 1 is a positive example for the class
            training    is a dictionary
            clabel         name of classifier
        '''
        # Added a label field, contains the name of the dataset
        self.clabel = clabel
        self.t_labels = training_labels
        self.training = training
        self.v_labels = validation_labels
        self.validation = validation
        self.tuning_log = None

        # Ranges for parameter C and gamma
        self.n_iterations = 1
        self.C_range = [pow(2,-5), pow(2,15)]            #max pow(2,15)
        self.C_step = float(self.C_range[1] - self.C_range[0])/self.n_iterations
        self.gamma_range = [pow(2,-15), pow(2,3)]       #max pow(2,3)
        self.gamma_step = float(self.gamma_range[1] - self.gamma_range[0])/self.n_iterations
        self.finer_range = { 'C': 100, 'gamma': 1} 


        ## SVM initialization
        # To disable output messages from the library
        svmc.svm_set_quiet()
        # Definition of standard parameters for SVM
        self.parameters = svm_parameter(svm_type = C_SVC, kernel_type = RBF, \
                C = 1, gamma = 0, probability = 1)
        # Definition of the problem wrt training examples
        self.problem = svm_problem(self.t_labels, self.training)


    def update_parameters(self, C, gamma):
        '''
            Function that update SVM parameters
        '''
        self.parameters.C = C
        self.parameters.gamma = gamma


    def train(self):
        '''
            Train the SVM defined by problem  with the parameters given
        '''
        self.model = svm_model(self.problem,self.parameters)


    def classify(self, sample):
        '''
            Given a sample in the form of k-gram, it retrieve the prediction
            and the probability that it can be in the class 1
            returns:     [ prediction, probability ]
        '''
        # Return: class_prediction + probability of prediction  
        prediction = self.model.predict_probability(sample)
        return prediction


    def validate(self):
        '''
            Consider the given validation dataset and perform the evaluation
            of the model.
        '''
        i = 0
        pred = {}
        correct = wrong = total = 0
        for d in self.validation:
            pred = self.classify(d)
            if pred[0] == self.v_labels[i]:
                correct += 1
            else:
                wrong += 1
            total += 1
            i += 1
        return correct, wrong, total


    def stochastic_search(self):
        print "Stochastic"

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
                self.parameters.C = C
                self.parameters.gamma = gamma
                self.model = svm_model(self.problem, self.parameters)
                line = "*** TUNING: C = %f; gamma = %f \n" % (C, gamma)
                self.log(line)
                c,w,t = self.validate()
                precision = float(c)/t
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
            start = [self.C_range[0], self.gamma_range[0]]
            end = [self.C_range[1], self.gamma_range[1]]
            step = [self.C_step, self.gamma_step]
        else:
            c_start = element[0] - self.finer_range['C']
            c_end = element[0] + self.finer_range['C']

            gamma_start = element[1] - self.finer_range['gamma']
            gamma_end = element[1] + self.finer_range['gamma']

            if c_start < self.C_range[0]:
                c_start = self.C_range[0]
            if c_end > self.C_range[1]:
                c_end = self.C_range[1]
            if gamma_start < self.gamma_range[0]:
                gamma_start = self.gamma_range[0]
            if gamma_end > self.gamma_range[1]:
                gamma_end = self.gamma_range[1]

            start = [c_start, gamma_start]
            end = [c_end, gamma_end]
        
        C_step = float(end[0] - start[0])/self.n_iterations
        gamma_step = float(end[1] - start[1])/self.n_iterations
        step = [C_step, gamma_step]
        return (start, end, step)

        
    def tuning(self, parameter):
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
        kernels = [LINEAR, POLY, RBF, SIGMOID]
        
        start, end, step = self.parameter_search(None, 0)
        print "PARAM", start, end, step
        line = "Validation on %s \nCoarse: C = [%f, %f], step %f, gamma = " + \
               "[%f,  %f], step %f \n"  % (self.clabel, float(start[0]),
               float(end[0]), float(step[0]),\
                float(start[1]), float(end[1]), float(step[1]))
        self.log(line)
        best_param, prec = self.iterative_tuner(start, end, step)
        line = "\nBest parameters found: C = %f, gamma = %f; The precision is " + \
               "about %f \n" % (float(best_param[0]), float(best_param[1]), prec)
        self.log(line)

        
        start, end, step = self.parameter_search(best_param, 1)
        line = "\n\n Validation on the finer range: C = [%f, %f], step %f, gamma = " + \
               "[%f, %f], step %f \n" % (float(start[0]), float(end[0]), float(step[0]), \
                float(start[1]), float(end[1]), float(step[1]))
        self.log(line)
        best_param, prec = self.iterative_tuner(start, end, step)
        print "STATUS: start: ", start, "  end: ", end, "  step: ",step
        print "ERROR: ", best_param
        line = "\n\n\nBest parameters found: C = %f, gamma = %f; The precision\ is " + \
               "about %f \n" % (best_param[0], best_param[1], prec)
        self.log(line)

        line  = "Setting up the model with the parameters: C = %f, gamma = %f"  \
                % (best_param[0], best_param[1])
        self.log(line)
        self.update_parameters(best_param[0], best_param[1])
        self.model = svm_model(self.problem, self.parameters)

        self.log(None)
        print line

    def log(self, data):
        '''
            Define the log function to store data in a file, default is the
            "tuning.log"
        '''
        if self.tuning_log == None:
            self.tuning_log = open("tuning.log",'w')

        if data == None:
            self.tuning_log.close()
        else:
            print "Write data ", data
            self.tuning_log.write(data)
