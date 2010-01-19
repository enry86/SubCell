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
        
        # Ranges for parameter C and gamma
        self.C_range = [pow(2,-5), pow(2,15)]            #max pow(2,15)
        self.C_step = 10
        self.gamma_range = [pow(2,-15), pow(2,3)]       #max pow(2,3)
        self.gamma_step = 0.710 
        self.finer_range = { 'C': 100, 'gamma': 0.1} 


        ## SVM initialization
        # To disable output messages from the library
        svmc.svm_set_quiet()
        # Definition of standard parameters for SVM
        self.parameters = svm_parameter(svm_type = C_SVC, kernel_type = RBF, \
                C = 1, gamma = 0, probability = 1)
        # Definition of the problem wrt training examples
        self.problem = svm_problem(self.t_labels, self.training)


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


    def iterative_tuner(self, start, end, step):
        '''
            Simple function to perform the validation in two passages, the
            first has a wide range and a big step, the second has a minor
            range and a little step
        '''
        C = start[0]
        while C <= end[0]:
            gamma = start[1]
            while gamma <= end[0]:
                self.parameters.C = C
                self.parameters.gamma = gamma
                print "*** TUNING: C = %f; gamma = %f" % (C, gamma)
                c,w,t = self.validate()
                precision = float(c)/t
                if precision > best:
                    best = precision
                    data = [C, gamma]
                gamma += step[1]
            C += step[0]
        return (data, best)

        
    def tuning(self, parameter):
        '''
            Considering default kernel as RBF - for the moment is the
            simplier.
            parameter:  if 0, then no paramters will be used;
                dicts of parameters:
                    Kernel  "stands for the Kernel type to use"
                    C       "stands for the C regulation parameter" 
                    gamma   "stands for the gamma kernel parameter"
            IMPROVEMENT: inserire parametri da riga di comando, filtrarli qui
            nei parametri
        '''
        # Need to iterate on the possible kernels
        kernels = [LINEAR, POLY, RBF, SIGMOID]
        # HOW TO PARALLELIZE ?
        if parameter == 0:
            C = self.C_range[0] 
            gamma = self.gamma_range[0]
       
        # for k in kernels:
        #       self.parameters.kernel = k

        res = {}
        i = 0
        data = []
        best = 0.

        #coarse

        #finer(C, gamma)

        while C <= self.C_range[1]:
            gamma = self.gamma_range[0]
            while gamma <= self.gamma_range[1]:
                self.parameters.C = C
                self.parameters.gamma = gamma
                self.model = svm_model(self.problem, self.parameters)

                print "*** TUNING: C = %f; gamma = %f" % (C, gamma)
                c,w,t = self.validate()
                precision = float(c)/t
                if precision > best:
                    best = precision
                    data = [C, gamma]

                print "Correct: %i / %i " % (c,t)
                res.__setitem__(i, [C, gamma, float(c)/t] )
                i += 1
                gamma += self.gamma_step
            C += self.C_step
        print res

        # Need to perform the last computation? (i.e. The one with C_range[1])



