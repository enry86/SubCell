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
        C_range = [pow(2,-5), pow(2,5)]            #max pow(2,15)
        C_step = 10
        gamma_range = [pow(2,-15), pow(2,-5)]       #max pow(2,3)
        gamma_step = 0.010 
        finer_range = { 'C': 100, 'gamma': 0.1} 

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
            if pred[0] == v_labels[i]:
                correct += 1
            else:
                wrong += 1
            total += 1
            i += 1
        return correct, wrong, total

        
    def tuning(self, parameter):
        '''
            Considering default kernel as RBF - for the moment is the
            simplier.
            parameter:  if 0, then no paramters will be used;
                dicts of parameters:
                    Kernel  "stands for the Kernel type to use"
                    C       "stands for the C regulation parameter" 
                    gamma   "stands for the gamma kernel parameter"
        '''
        if param == 0:
            C = C_range[0] 
            gamma = gamma_range[0]
        
        while C <= C_range[1]:
            while gamma <= gamma_range[1]:
                self.parameters.C = C
                self.parameters.gamma = gamma
                    
                gamma += gamma_step
            C += C_step

        # Need to perform the last computation? (i.e. The one with C_range[1])


    def classify_ds(self, ds, n):
        '''
            classifies the test dataset, return the number of correct and
            wrong predictions and the nuber of total instances tested
        '''
        total = 0
        corr = 0
        wrong = 0
        for d in ds:
            pred = self.classify(d)
            for p in preds:
                if preds[p][1][1] > best:
                    best = preds[p][1][1]
                    cls = p
            if cls == n:
                corr += 1
            else:
                wrong += 1
            total += 1
        return (corr, wrong, total)
        
    def evaluate_model(self, ds_ext):
        res = {}
        ds = open('.tmp/' + clabel + ds_ext, 'r')
        ds_l = self.read_ds(ds, s)
        ds.close()
        res = self.classify_ds(ds_l, clabel)
        return res

    def test_model(self):
        result =  evaluate_model('.tst')
        return result

    def validate_model(self):
        result = evaluate_model('.val')
        return result
