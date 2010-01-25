# -*- coding: utf-8 -*-
'''Classifier functions'''

from svm import *
import math
import os
import tuner

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
#        self.tuning_log = None

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
                C = self.C_range[0], gamma = self.gamma_range[0], probability = 1)
        # Definition of the problem wrt training examples
        self.problem = svm_problem(self.t_labels, self.training)

        # Initializing the external tuner
        self.tune = tuner.Tuner(self)


    def update_parameters(self, C, gamma):
        '''
            Function that update SVM parameters
        '''
        if C != None:
            self.parameters.C = C
        if gamma != None:
            self.parameters.gamma = gamma

    def train(self):
        '''
            Train the SVM defined by problem  with the parameters given
        '''
        self.svm_check_parameter()
        self.model = svm_model(self.problem,self.parameters)


    def classify(self, sample):
        '''
            Given a sample in the form of k-gram, it retrieve the prediction
            and the probability that it can be in the class 1
            returns:     [ prediction, probability ]
        '''
        prediction = self.model.predict_probability(sample)
        return prediction


    def validate(self):
        '''
            Consider the given validation dataset and perform the evaluation
            of the model.
        '''
        i = 0
        pred = {}
        correct = wrong = nr = total = 0
        for d in self.validation:
            pred = self.classify(d)
            if (pred[0] == 1) and (self.v_labels[i] == 1):
                correct += 1
            elif not pred[0] and (self.v_labels[i] == 1):
                nr += 1
            elif self.v_labels[i] == 1:
                wrong += 1
            total += 1
            i += 1
        return correct, wrong, nr, total


    def tuning(self, parameter):
        '''
            Considering default kernel as RBF.
            parameter:  if 0, then no parameters will be used;
                dicts of parameters:
                    C       "stands for the C regulation parameter" 
                    gamma   "stands for the gamma kernel parameter"
        '''
        # Need parametrization
        self.tune.tune(parameter)
