'''Classifiers execution manager'''

import classifier
import time
import threading
import measure

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
        self.tst = []
        t_lab = []
        v_lab = []
        self.s_lab = []
        for n in self.names:
            ds_t = open('.tmp/' + n + '.trn')
            ds_v = open('.tmp/' + n + '.val')
            ds_s = open('.tmp/' + n + '.tst')
            tmp_d, tmp_l = self.read_ds(ds_t, n)
            trn += tmp_d
            t_lab += tmp_l
            tmp_d, tmp_l = self.read_ds(ds_v, n)
            val += tmp_d
            v_lab += tmp_l
            tmp_d, tmp_l = self.read_ds(ds_s, n)
            self.tst += tmp_d
            self.s_lab += tmp_l
            ds_t.close()
            ds_v.close()
            ds_s.close()
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
    

    def read_ds(self, ds, n):
        '''
            Utility for reading a dataset form file
        '''
        res = []
        lab = []
        for l in ds:
            if l[0] != '>' and l != '\n':
                res.append(self.sker.to_vector(l))
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


    def test(self):
        i = 0
        self.meas = measure.Measure(self.names)
        for s in self.tst:
            c, r = self.class_sample(s)
            self.meas.update_res(r, self.s_lab[i]) 
            if c == self.s_lab[i]:
                cls = True
            else:
                cls = False
            self.meas.update_count(cls, self.s_lab[i])
            i += 1
        return self.meas


    def class_sample(self, sam):
        best = 0.0
        cls = ''
        res = {}
        for s in self.svms:
            res[s.clabel] = s.classify(sam)
        for r in res:
            if res[r][1][1] > best:
                best = res[r][1][1]
                cls = r
        return cls, res


    def validation(self, parameters):
        '''
            Manages the execution of the validation session, performing the
            tuning for all the svm passing the common fixed set of paramters.
        '''
        for svm in self.svms:
            svm.tuning(parameters)
