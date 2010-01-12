'''Classifier functions'''

from svm import *

class Classifier:
    def __init__(self, labels, training, validation, clabel):
        '''
            Initialization of the problem through training data.
            labels      is a list of 0,1 that recognize each sample in the
                        training, where 1 is a positive example for the class
            training    is a dictionary
            clabel         name of classifier
        '''
        # Added a label field, contains the name of the dataset
        self.clabel = clabel
        self.labels = labels
        self.training = training
        self.validation = validation
        # To disable output messages from the library
        svmc.svm_set_quiet()
        # Definition of standard parameters for SVM
        self.parameters = svm_parameter(C = 100, kernel_type = RBF, probability \
                = 1)
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
        prediction = self.model.predict_probability(sample)
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

