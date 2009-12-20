'''Classifier functions'''

from svm import *

class Classifier:
    def __init__(self, labels, training, validation):
        '''
            Initialization of the problem through training data.
            labels      is a list of 0,1 that recognize each sample in the
                        training, where 1 is a positive example for the class
            training    is a dictionary
        '''
        self.labels = labels
        self.training = training
        self.validation = validation
        # To disable output messages from the library
        svmc.svm_set_quiet()
        # Definition of standard parameters for SVM
        self.parameters = svm_parameter(C = 1, kernel_type = LINEAR)
        # Definition of the problem wrt training examples
        self.problem = svm_problem(self.labels, self.training)

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
        prediction = []
        prediction[0], prediction[1] = self.model.predict_probability(sample)
        return prediction


    def validation(self, param):
        '''
            param:  if 0, then no paramters will be used;
                    dicts of paramters:
                        C   "stands for the C regulation paramter" 
        '''

    def tuning(self, parameter):
        '''
            paramter:   paramters t tweak
        '''

