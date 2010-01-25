'''
    Tools and utils for metrics evaluation
'''

class Measure:
    '''
        Class which handles metrics computation
    '''
    
    def __init__(self, names):
        self.res = {}
        self.cnt = {}
        for n in names:
            self.res[n] = [0,0,0]
            self.cnt[n] = [0,0]
        

    def update_res(self, preds, c):
        '''
            Updates the count of the results for a single svm
            pred    prediction computed by SVM
            c       true class of the sample

            res[name][0] => correct
            res[name][1] => mistake
            res[name][2] => non retrieved
        '''
        for p in preds:
            if p == c:
                if preds[p][0] == 1.0:
                    self.res[p][0] += 1
                else:
                    self.res[p][2] += 1
            else:
                if preds[p][0] == 1.0:
                    self.res[p][1] += 1
    

    def update_count(self, res, cls):
        '''
            Update counters for the global classification
        '''
        if res == True:
            self.cnt[cls][0] += 1
        self.cnt[cls][1] += 1


    def ds_counter(self, ds):
        '''
            Returns the counter for a specific dataset as a list:
            [correct, total]
        '''
        return self.cnt[ds]


    def all_counter(self):
        '''
            Computes the overall results for the global classifier
            Returns a list: [correct, total]
        '''
        c = 0
        t = 0
        for n in self.cnt:
            c += self.cnt[n][0]
            t += self.cnt[n][1]
        return [c, t]


    def fmeasure(self, p, r):
        '''
            Computes f measure given precision and recall
            p   precision
            r   recall
        '''
        try:
            res = 2.0 * (p * r) / (p + r)
        except:
            res = 0.0
        return res


    def precision(self, res):
        '''
            Computes precision for the given result list
            res[0]  correct
            res[1]  mistake
            res[2]  non ret
        '''
        try:
            prec = res[0] / float(res[0] + res[1])
        except:
            prec = 1.0
        return prec


    def recall(self, res):
        '''
            Computes recall on the given results list
            res[0]  correct
            res[1]  mistake
            res[2]  non ret
        '''
        try:
            rec = res[0] / float(res[0] + res[2])
        except:
            rec = 0.0
        return rec


    def svm_metrics(self, svm):
        '''
            Computes quality measures on the results stored in self.res
            svm     name of the svm

            returns a tuple: (precision, recall, f-measure)
        '''
        pr = self.precision(self.res[svm])
        re = self.recall(self.res[svm])
        fm = self.fmeasure(pr, re)
        return (pr, re, fm)


    def all_metrics(self):
        '''
            Groups the metrics for all the svms, storing them into a
            dictionary
        '''
        met = {}
        for r in self.res:
            met[r] = self.svm_metrics(r)
        return met


    def micro_average(self):
        '''
            Computes the microaverage of the three measures for the global
            classifier
            Returns a tuple: (mu-precision, mu-recall, mu-fmeasure)
        '''
        sums = [0,0,0]
        for r in self.res:
            for v in range(3):
                sums[v] +=  self.res[r][v]
        pr = self.precision(sums)
        re = self.recall(sums)
        fm = self.fmeasure(pr, re)
        return (pr, re, fm)
        
        
    
        
