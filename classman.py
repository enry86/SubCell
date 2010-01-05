'''Classifiers execution manager'''

import classifier
import time
import threading

class ClassMan:
    '''
        Manages the execution of the classifier
    '''
    
    def __init__(self, sker, names):
        '''
            # sker      string kernel list
            # names     list of class names
        '''
        self.sker = sker
        self.names = names


    def init_ds(self):
        '''
            reads the training and validation datasets from files and
            stores them in a list
        '''
        trn = []
        val = []
        lab = []
        for k in self.sker:
            n = k.lab
            ds_t = open('.tmp/' + n + '.trn')
            ds_v = open('.tmp/' + n + '.val')
            for l in ds_t:
                if l[0] != '>' and l != '\n':
                    trn.append(k.to_vector(l))
                    lab.append(n)
            val += self.read_ds(ds_v, k)
            ds_t.close()
            ds_v.close()
        return (lab, trn, val)


    
    def get_lab(self, name, lab):
        '''
            computes the list of labels for libsvm
        '''
        res = []
        for l in lab:
            if name == l:
                res.append(1)
            else:
                res.append(0)
        return res
    

    def read_ds(self, ds, k):
        '''
            Utility for reading a dataset form file
        '''
        res = []
        for l in ds:
            if l[0] != '>' and l != '\n':
                res.append(k.to_vector(l))
        return res
        

    def init_classifier(self):
        '''
            SVM initialization, stores the list of SVM objects in a list
            at instance level
        '''
        svms = []
        print '\nSVM initialization:'
        startt = time.time()
        lab, trn, val = self.init_ds()
        for n in self.names:
            tmp = classifier.Classifier(self.get_lab(n, lab), trn, val, n)
            print '\tSVM for %s initialized' % n
            svms.append(tmp)
        print 'Classifier initialized in %s sec.\n' % \
            (time.time() - startt)
        self.svms = svms

   
    def train(self, mt):
        '''
            wrapper for training SVMs
        '''
        print 'Training SVM:'
        if mt:
            self.multi_train()
        else:
            self.single_train()


    def single_train(self):
        '''
            triggers the training of the SVMs, single threaded
        '''
        startt = time.time()
        for s in self.svms:
            s.train()
            print '\tSVM for %s trained' % s.lab
        print 'SVM trained in %s sec. (single thread)\n' %  \
            (time.time() - startt)


    def multi_train(self):
        '''
            triggers the training of the SVMs, with a multithreadeing
            flavour
        '''
        startt = time.time()
        for s in self.svms:
            tmp = threading.Thread(target = s.train)
            tmp.start()
            print "Starting train"
        print 'SVM trained in %s sec. (multithreading)\n' %  \
            (time.time() - startt)

    
    def classify_ds(self, ds, n):
        '''
            classifies the test dataset, return the number of correct and
            wrong predictions and the nuber of total instances tested
        '''
        total = 0
        corr = 0
        wrong = 0
        for d in ds:
            preds = {}
            best = 0.0
            cls = ''
            for s in self.svms:
                preds[s.lab] = s.classify(d)
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


    def precision(self, res):
        '''
            computes the precision of the classification given the result
            tuple
        '''
        tot = 0
        cor = 0
        for r in res:
            tot += res[r][2]
            cor += res[r][0]
        return float(cor) / tot


    def test(self):
        '''
            Manages the execution of a test session
        '''
        res = {}
        print 'Performing test:'
        for s in self.sker:
            n = s.lab
            ds = open('.tmp/' + n + '.tst', 'r')
            ds_l = self.read_ds(ds, s)
            ds.close()
            res[n] = self.classify_ds(ds_l, n)
            print '\t', n, ':', res[n][0], '/', res[n][2]
        pr = self.precision(res)
        print 'Precision on test dataset =', pr, '\n'

   
