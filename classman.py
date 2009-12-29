'''Classifiers execution manager'''

import classifier
import time
import threading

class ClassMan:
    def __init__(self, sker, names):
        self.sker = sker
        self.names = names


    def init_ds(self):
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
            for l in ds_v:
                 if l[0] != '>' and l != '\n':
                    val.append(k.to_vector(l))
        return (lab, trn, val)

    
    def get_lab(self, name, lab):
        res = []
        for l in lab:
            if name == l:
                res.append(1)
            else:
                res.append(0)
        return res


    def init_classifier(self):
        svms = []
        startt = time.time()
        lab, trn, val = self.init_ds()
        for n in self.names:
            tmp = classifier.Classifier(self.get_lab(n, lab), trn, val)
            print 'SVM for %s initialized' % n
            svms.append(tmp)
        print 'Classifier initialized in %s sec.' % (time.time() - startt)
        return svms

   
    def train(self, svms, mt):
        if mt:
            self.multi_train(svms)
        else:
            self.single_train(svms)


    def single_train(self, svms):
        startt = time.time()
        for s in svms:
            s.train()
        print 'SVM trained in %s sec. (single thread)' % (time.time() - startt)


    def multi_train(self, svms):
        startt = time.time()
        for s in svms:
            tmp = threading.Thread(target = s.train)
            tmp.start()
            thr.append(tmp)
        print 'SVM trained in %s sec. (multithreading)' % (time.time() - \
            startt)

