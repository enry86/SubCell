'''Classifier core functions and classes'''

class Classifier:
    def __init__(self, ds, k, lab):
        self.kgr = self.retrieve_sub(ds, k)
        self.k = k
        self.lab = lab

    def retrieve_sub(self, ds, k):
        res = {}
        for l in ds:
            if l[0] != '>' and l != '\n':
                self.add_sub(res, l, k)
        ds.close()
        return res

    def add_sub(self, kgr, l, k):
        for i in range(len(l)):
            tmp = l[i : i + k]
            if len(tmp) == k:
                if kgr.has_key(tmp):
                    kgr[tmp] += 1
                else:
                    kgr[tmp] = 1


