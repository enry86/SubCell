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
            val += self.read_ds(ds_v, k)
            ds_t.close()
            ds_v.close()
        return (lab, trn, val)


    
    def get_lab(self, name, lab):
        res = []
        for l in lab:
            if name == l:
                res.append(1)
            else:
                res.append(0)
        return res
    

    def read_ds(self, ds, k):
        res = []
        for l in ds:
            if l[0] != '>' and l != '\n':
                res.append(k.to_vector(l))
        return res
        

    def init_classifier(self):
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
        print 'Training SVM:'
        if mt:
            self.multi_train()
        else:
            self.single_train()


    def single_train(self):
        startt = time.time()
        for s in self.svms:
            s.train()
            print '\tSVM for %s trained' % s.lab
        print 'SVM trained in %s sec. (single thread)\n' %  \
            (time.time() - startt)


    def multi_train(self):
        startt = time.time()
        for s in self.svms:
            tmp = threading.Thread(target = s.train)
            tmp.start()
        print 'SVM trained in %s sec. (multithreading)\n' %  \
            (time.time() - startt)

    
    def classify_ds(self, ds, n):
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
        tot = 0
        cor = 0
        for r in res:
            tot += res[r][2]
            cor += res[r][0]
        return float(cor) / tot


    def test(self):
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

   
