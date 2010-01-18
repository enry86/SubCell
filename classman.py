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
        t_lab = []
        v_lab = []
        for k in self.sker:
            n = k.lab
            ds_t = open('.tmp/' + n + '.trn')
            ds_v = open('.tmp/' + n + '.val')
            tmp_d, tmp_l = self.read_ds(ds_t, k, n)
            trn += tmp_d
            t_lab += tmp_l
            tmp_d, tmp_l = self.read_ds(ds_v, k, n)
            val += tmp_d
            v_lab += tmp_l
            ds_t.close()
            ds_v.close()
        return (t_lab, trn, v_lab, val)


    
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
    

    def read_ds(self, ds, k, n):
        '''
            Utility for reading a dataset form file
        '''
        res = []
        lab = []
        for l in ds:
            if l[0] != '>' and l != '\n':
                res.append(k.to_vector(l))
                lab.append(n)
        return (res, lab)
        

    def init_classifier(self):
        '''
            SVM initialization, stores the list of SVM objects in a list
            at instance level
        '''
        svms = []
        print '\nSVM initialization:'
        startt = time.time()
        t_lab, trn, v_lab, val = self.init_ds()
        for n in self.names:
            tmp = classifier.Classifier(self.get_lab(n, t_lab), trn, \
                    self.get_lab(n, v_lab), val, n)
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
            print '\tSVM for %s trained' % s.clabel
        print 'SVM trained in %s sec. (single thread)\n' %  \
            (time.time() - startt)


    def multi_train(self):
        '''
            triggers the training of the SVMs, with a multithreadeing
            flavour
        '''
        thrs = []
        startt = time.time()
        for s in self.svms:
            tmp = threading.Thread(target = self.start_train, \
                args = (s,))
            thrs.append(tmp)
            tmp.start()
            print '\tTraining for %s started...' % s.clabel
        for t in thrs:
            t.join()
        print 'SVM trained in %s sec. (multithreading)\n' %  \
            (time.time() - startt)

    
    def start_train(self, s):
        '''
            starts the training task, inserts a little gap in
            order to permit the correct starting of the thread
        '''
        time.sleep(0.2)
        s.train()


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
                preds[s.clabel] = s.classify(d)
            for p in preds:
                try:
                    if preds[p][1][1] > best:
                        best = preds[p][1][1]
                        cls = p
                except KeyError:
                    pass
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

        POSSIBILE RENDERLA COME VALIDATION SOTTO?
        '''
        res = {}
        print 'Performing test:'
        for s in self.sker:
            n = s.lab
            ds_file = open('.tmp/' + n + '.tst', 'r')
            ds, ds_l = self.read_ds(ds_file, s, n)
            ds_file.close()
            res[n] = self.classify_ds(ds, n)
            print '\t', n, ':', res[n][0], '/', res[n][2]
        pr = self.precision(res)
        print 'Precision on test dataset =', pr, '\n'


    def validation(self):
        '''
            Manages the execution of the validation session

        MODIFICARE INSIEME A TEST...ENTRAMBE SONO INERENTI A SVMS QUINDI AI 4
        SVM INSIEME
        '''
        res = {}
        for svm in self.svms:
            n = svm.clabel
            res[n] = svm.validate()
            print 't', n, ':', res[n][0], '/', res[n][2]
        pr = self.precision(res)
        print 'Precision on validation dataset =', pr, '\n'

