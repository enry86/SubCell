'''String Kernel core functions'''

class StrKernel:
    
    def __init__(self, ds_n, ds_d, k_list):
        '''
            # ds_n      filenames of the dataset
            # ds_d      directory of the dataset
            # k_list    list of k-gram dimension
        '''
        
        self.k_list = k_list
        files = []
        for d in ds_n:
            files.append(open(ds_d + d + '.trn','r'))
        tmp = self.retrieve_sub(files)
        self.kgr = self.build_hash(tmp)
        for f in files:
            f.close()



    def retrieve_sub(self, ds_f):
        '''
            retrieves all the substrings in the dataset
        '''
        sub = {}        # Stored as a dictionary for speed reasons 
        for f in ds_f:
            self.read_file(f, sub)
        res = sub.keys()
        res.sort()
        return res      # Returned a list of keys, sorted

    def read_file(self, f, sub):
        for l in f:
            if l[0] != '>' and l != '\n':
                self.add_sub(sub, l)

    def add_sub(self, kgr, l):
        '''
            fills a dictionary with substrings count of a single protein
        '''
        for k in self.k_list:
            for i in range(len(l[:-1]) - (k - 1)):
                tmp = l[i : i + k]
                if kgr.has_key(tmp):
                    kgr[tmp] += 1
                else:
                    kgr[tmp] = 1
    

    def build_hash(self, kgr):
        res = {}
        for i in range(len(kgr)):
            res[kgr[i]] = i
        return res


    def to_vector(self, p):
        '''
            computes the sparse vector representing a protein storing it
            in a dictionary
        '''
        res = {}
        tmp = {}
        self.add_sub(tmp, p)
        for k in tmp:
            if self.kgr.has_key(k):
                res[self.kgr[k]] = tmp[k]
        return res


