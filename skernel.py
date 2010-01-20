'''String Kernel core functions'''

class StrKernel:
    
    def __init__(self, ds, k_list, lab):
        '''
            # ds        files of the dataset
            # k_list    list of k-gram dimension
            # lab       list of labels
        '''
        self.kgr = self.retrieve_sub(ds, k_list)
        self.k_list = k_list
        self.lab = lab


    def retrieve_sub(self, ds, k_list):
        '''
            retrieves all the substrings in the dataset
        '''
        sub = {}        # Stored as a dictionary for speed reasons 
        for l in ds:
            if l[0] != '>' and l != '\n':
                self.add_sub(sub, l, k_list)
        res = sub.keys()
        res.sort()
        return res      # Returned a list of keys, sorted


    def add_sub(self, kgr, l, k_list):
        '''
            fills a dictionary with substrings count of a single protein
        '''
        for k in k_list:
            for i in range(len(l[:-1]) - (k - 1)):
                tmp = l[i : i + k]
                if kgr.has_key(tmp):
                    kgr[tmp] += 1
                else:
                    kgr[tmp] = 1


    def to_vector(self, p):
        '''
            computes the sparse vector representing a protein storing it
            in a dictionary
        '''
        res = {}
        tmp = {}
        self.add_sub(tmp, p, self.k_list)
        for i in range(len(self.kgr)):
            if tmp.has_key(self.kgr[i]):
                res[i] = tmp[self.kgr[i]]
        return res


