#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:51:21 2018

@author: siting.liang
"""

import numpy as np
import spacy
nlp=spacy.load("en_core_web_lg")
#import sys
from pickle_save import save
from text2sents import sense2vec_transform, transform_sentences_ngram
from gensim.models import KeyedVectors
#vocab=set()
#vocab_bi=set()
#vocab_sense=set()

vec_dict={0:"tokens_uni.vec",1:"tag_dep_.vec",2:"tag_dep_head.vec",3:"cluster_uni.vec"}
bi_grams=["tokens_bi.vec","tags_bi.vec"]
sense_vec="sense2vec.vec"
en_wiki="wiki.en.vec"


#def get_vec(path):
    #keyvectors=KeyedVectors.load_word2vec_format(path,binary=False)
    #vocab=keyvectors.vocab
    #vectors=[keyvectors.get_vector(w) for w in vocab]
    #return vocab, vectors

def transform_sent(text):
    doc=nlp(text)
    new_sent=[(t.text,t.tag_+"_"+t.dep_,t.tag_+"_"+t.head.tag_+"_"+t.dep_+"_"+t.head.dep_,str(t.cluster)) for s in doc.sents for t in s]
    return new_sent

def process_doc(line):
    new_line=line.replace("http","").replace("www","").replace(":", " ").replace("/"," ").replace("."," ").replace("(","").replace(")","")
    
    #for w in new_line.split():
        #vocab.add(w)
    
    new_doc=transform_sent(new_line)
    
    return new_doc

def get_vocab_embed(w,keyvectors):
    
    #vocab=[v for v in keyvectors.vocab]
    #vectors=[]
    #for w in words:
    if w in keyvectors.vocab:
           vector=keyvectors.get_vector(w)
    else:
        vector=np.zeros(300)
    #print(len(vocab),type(vocab))
    return vector   

def load_results(result, index=None, bi_gramm=False):
    
    result_dict={}
    
    if index == None:
        path=en_wiki
        keyvectors=KeyedVectors.load_word2vec_format(path,binary=False)
    elif len(index)==1: 
        path= vec_dict[0]
        keyvectors=KeyedVectors.load_word2vec_format(path,binary=False)
    elif len(index) > 1:
        paths=[vec_dict[i] for i in index ]
        keyvectors=[KeyedVectors.load_word2vec_format(path,binary=False) for path in paths ]
    if bi_gramm==True:
       keyvectors=[]
        
        
        
    with open(result,"r") as file:
        for line in file.readlines()[1:]:
             l=[]
             new_line=line.split()
             #print(new_line)
             IDs = new_line[0].split(".")
             #print(IDs)
             ID=IDs[0]
             new_sent=process_doc(" ".join(new_line[1:]))
             for w in new_sent:
                 if index == None:
                    l.append(get_vocab_embed(w[0],keyvectors))
                 elif  index == 0 :
                        l.append(get_vocab_embed(w[0],keyvectors))
                 elif index > 1 :
                     ws=[w[i] for i in index]
                     for wt in ws :
                         for n in range(len(index)):
                            l.append(get_vocab_embed(wt[n],keyvectors[n]))
             
             
             if ID not in result_dict:
                 result_dict[ID]=[]
                 result_dict[ID].append(l)
             else:
                 result_dict[ID].append(l)
    
    return result_dict



"""
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
"""


def make_sense2vec_embed(result):
    result_embed={}
    result_vocab={}
    #keyvectors=KeyedVectors.load_word2vec_format(sense_vec,binary=False)
    with open(result,"r") as file:
        for line in file.readlines()[1:]:
             #l=[]
             s=[]
             new_line=line.split()
             #print(new_line)
             IDs = new_line[0].split(".")
             #print(IDs)
             ID=IDs[0]
             new_doc=sense2vec_transform(nlp(" ".join(new_line[1:])))
             for s in new_doc:
                 for w in s:
                     #l.append(get_vocab_embed(w,keyvectors))
                     s.append(w)
             #if ID not in result_embed:
                 #result_embed[ID]=[]
                 #result_embed[ID].append(l)                          
             #else:
                 #result_embed[ID].append(l)
                 
                 
             if ID not in result_vocab:
                 result_vocab[ID]=[]
                 result_vocab[ID].append(s)
             else:
                result_vocab[ID].append(s)
    return  result_vocab, result_embed

result="/Users/siting.liang/Documents/Semantik/WSI-Evaluator/datasets/MORESQUE/results.txt"   
sense2vec_vocab=make_sense2vec_embed(result)[0]  
save("sense2vec_vocab", sense2vec_vocab)




    
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
#line="http://en.wikipedia.org/wiki/The_Block_(album)	The Block (album) - Wikipedia, the free encyclopedia	The Block was released on September 2, 2008 and debuted at number one on the ... New Kids on the Block · Hangin' Tough · Merry, Merry Christmas · Step by Step ..."
#print(process_doc(line))
#print(vocab)
#if "__name__"=="__main__":
    #path=sys.argv[1]
    #path="/datasets/MORESQUE/results.txt"
   # res_dict=read_results(path)
    #print(len(res_dict))
    #print(res_dict["45"])
#path_dict="/Users/siting.liang/Documents/Semantik/WSI-Evaluator/datasets/MORESQUE/result_dict.pickle"
#path_vocab="/Users/siting.liang/Documents/Semantik/WSI-Evaluator/datasets/MORESQUE/vocab.pickle"
#save(path_dict, res_dict)
#save(path_vocab,vocab)




#read_results(result)
