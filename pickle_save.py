#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 09:48:36 2018

@author: siting.liang
"""

import pickle

#path_vocab="/Users/siting.liang/Documents/CRMbot/vocab_embed.txt"
#path_all="/Users/siting.liang/Downloads/wiki.de.vec"


class MacOSFile(object):
    def __init__(self, f):
        self.f=f
    
    def __getattr__(self, item):
        return getattr(self.f, item)
    
    def read(self, n):
        if n >=(1<<31):
            buffer = bytearray(n)
            idx=0
            while idx < n:
                batch_size=min(n-idx, 1 << 31 - 1)
                print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size]=self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)
    
    def write(self,buffer):
        n=len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx=0
        while idx < n:
            batch_size = min(n-idx, 1<<31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx+=batch_size

def save(path, obj):  
    with open(path,"wb") as p_out:
         return pickle.dump(obj, MacOSFile(p_out), protocol=pickle.HIGHEST_PROTOCOL)
         
def load(pickle_in):
    with open(pickle_in, "rb") as f:
    
         return pickle.load(MacOSFile(f))


#all_w, all_e=get_embed(path_all,300)
#path_w="all_w.pickle"
#path_e="all_e.pickle"
#save(path_w,all_w)
#save(path_e,all_e)

