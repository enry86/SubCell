# -*- coding: utf-8 -*-
'''Classifier functions'''

from svm import *
import math
import os
import tuner
import measure

class Classifier:
    def __init__(self, training_labels, training, validation_labels, validation,\
                    clabel, C, gamma, iterations, penalty):
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
        self.n_iterations = iterations
        self.measure = measure.Measure([self.clabel])
        self.penalty = penalty

        # Ranges for parameter C and gamma
        # If C is None, the iteration on the range will be ignored
                
        # Check the C parameter
        if C != None:
            if C == 0:
                C = pow(2,-15)
            self.C_range = [C, C]
            self.C_step = self.C_range[0]
        else:
            self.C_range = [pow(2,-5), pow(2,15)]           #max pow(2,15)
            self.C_step = \
                    float(self.C_range[1] - self.C_range[0])/self.n_iterations
            self.C = self.C_range[0]

        # Check the gamma parameter
        if gamma != None:
            if gamma == 0:
                gamma = pow(2,-30)
            self.gamma_range = [gamma, gamma]
            self.gamma_step = self.gamma_range[0]
        else:
            self.gamma_range = [pow(2,-15), pow(2,3)]       #max pow(2,3)
            self.gamma_step = \
                    float(self.gamma_range[1] - self.gamma_range[0])/self.n_iterations
            self.gamma = self.gamma_range[0]
        self.finer_range = { 'C': 100, 'gamma': 1} 


        #print "C = [%f, %f]; gamma = [%f, %f]" % \
        #    (self.C_range[0],self.C_range[1],self.gamma_range[0],self.gamma_range[1])

        ## SVM initialization
        # To disable output messages from the library
        svmc.svm_set_quiet()
        # Definition of standard parameters for SVM
        self.parameters = svm_parameter(svm_type = C_SVC, kernel_type = RBF, \
                C = self.C_range[0], gamma = self.gamma_range[0], probability = 1, \
                nr_weight = 2, weight_label = [1, 0], weight = [1,1])
        # Definition of the problem wrt training examples
        self.problem = svm_problem(self.t_labels, self.training)

        # Initializing the external tuner
        self.tune = tuner.Tuner(self)


    def enable_penalty(self):
        '''
            Setting up the penalty for the negative examples
        '''
        self.parameters.weight = [1, self.penalty]
        print "Penalty for negative samples set to ", self.penalty


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
            prediction = self.classify(d)
            pred = {self.clabel: prediction}
            if self.v_labels[i] == 1:
                self.measure.update_res(pred, self.clabel)
                if prediction[0] == 1:
                    self.measure.update_count(True, self.clabel)
                else:
                    self.measure.update_count(False, self.clabel)
            else:
                self.measure.update_res(pred, '§')
                self.measure.update_count(False, self.clabel)
            i += 1


    def tuning(self):
        '''
            Considering default kernel as RBF.
            The function is supported by the class Tuner which search the best
            couple of C and gamma parameters.
            It will maximize the F-measure of the SVM.
        '''
        self.tune.tune()
