'''String Kernel core functions'''

class StrKernel:
    def __init__(self, ds, k_list, lab):
        self.kgr = self.retrieve_sub(ds, k_list)
        self.k_list = k_list
        self.lab = lab


    def retrieve_sub(self, ds, k_list):
        sub = {}        # Stored as a dictionary for speed reasons 
        for l in ds:
            if l[0] != '>' and l != '\n':
                self.add_sub(sub, l, k_list)
        res = sub.keys()
        res.sort()
        return res      # Returned a list of keys, sorted


    def add_sub(self, kgr, l, k_list):
        for k in k_list:
            for i in range(len(l) - (k - 1)):
                tmp = l[i : i + k]
                if kgr.has_key(tmp):
                    kgr[tmp] += 1
                else:
                    kgr[tmp] = 1


    def to_vector(self, p):
        res = {}
        tmp = {}
        self.add_sub(tmp, p, self.k_list)
        for i in range(len(self.kgr)):
            if tmp.has_key(self.kgr[i]):
                res[i] = tmp[self.kgr[i]]
        return res

