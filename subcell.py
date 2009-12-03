#!/usr/bin/python

import sys

def main():
    ds_n = sys.argv[1]
    ds = open(ds_n)
    k = 3 
    d = {}
    for i in ds:
        if i[0] != '>' and i != '\n':
            retrieve_subs(i, k, d)
    ds.close()
#    print d
    print len(d)

def retrieve_subs(str, k, d):
    for i in range(len(str)):
        tmp = str[i:i+k]
        if len(tmp) == k and tmp != '\n':
            if d.has_key(tmp):
                d[tmp] += 1
            else:
                d[tmp] = 1
        
        


if __name__ == "__main__":
    main()
