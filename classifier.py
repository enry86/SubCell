'''Classifier core functions and classes'''

class Classifier:
    def __init__(self, ds, k, lab):
        self.kgr = self.retrieve_sub(ds, k)
        self.k = k
        self.lab = lab

    def retrieve_sub(self, ds, k):
        sub = {}        # Stored as a dictionary for speed reasons 
        for l in ds:
            if l[0] != '>' and l != '\n':
                self.add_sub(sub, l, k)
        ds.close()
        res = sub.keys()
        res.sort()
        return res      # Returned a list of keys, sorted

    def add_sub(self, kgr, l, k):
        for i in range(len(l)):
            tmp = l[i : i + k]
            if len(tmp) == k:
                if not kgr.has_key(tmp):
                    kgr[tmp] = True

    def to_vector(self, p):
        res = []
        for k in self.kgr:
            res.append(p.count(k))
        return res

    def to_vector2(self, p):
        res = []
        tmp = {}
        self.add_sub(tmp, p, self.k)
        for i in self.kgr:
            if tmp.has_key(i):
                res.append(i)
        return res
        
            
