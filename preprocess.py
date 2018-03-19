#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:51:21 2018

@author: siting.liang
"""

import numpy as np
import spacy
nlp=spacy.load("en_core_web_lg")
import sys

def read_results(result):
    
    result_dict={}
    with open(result,"r") as file:
        for line in file.readlines()[1:]:
             
             new_line=line.split()
             #print(new_line)
             IDs = new_line[0].split(".")
             #print(IDs)
             ID=IDs[0]
             l=(IDs[1],new_line[1:])
             if ID not in result_dict:
                 result_dict[ID]=[]
                 result_dict[ID].append(l)
             else:
                 result_dict[ID].append(l)
    
    return result_dict

def s2float(line):
    n_line=np.asarray([float(l) for l in line ])
    return n_line

class embed_iter(object):
      def __init__(self,path):
          self.path=path
      def __iter__(self):
          with open(self.path, "r") as file:
              for line in file.readlines()[1:]:
                  new_l=line.split()
                  if len(new_l) == 301:
                      yield (new_l[0],s2float(new_l[1:]))
                  

   
def featurize_token(doc,embed,dim):
    vec={e[0]:e[1] for e in embed}
    w2vec=[]
    for w in doc:
        if w in vec:
            w2vec.append(vec[w])
        else:
            w2vec.append(np.zeros(dim))
    
    doc2v=np.array(w2vec) 
    return np.mean(doc2v,axis=0)


#def make_doc(list_):
    
    #or w in list_[1:] 


#def load_embed(embed_model):
#embed_path="/Users/siting.liang/Documents/Semantik/wiki.en/wiki.en.vec"
#embed=embed_iter(embed_path)   
#path="/Users/siting.liang/Documents/Semantik/WSI-Evaluator/datasets/MORESQUE/results.txt"
#res_dict=read_results(path)
#dim = 300
#print(len(res_dict))
#list_=res_dict["45"][0][1]
#doc2vec=featurize_token(list_[1:],embed,dim)
#print(doc2vec,len(doc2vec))


#if "__name__"=="__main__":
    path=sys.argv[1]
    #path="/datasets/MORESQUE/results.txt"
   # res_dict=read_results(path)
    #print(len(res_dict))
    #print(res_dict["45"])
    







#read_results(result)
