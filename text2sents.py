#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:27:23 2018

@author: siting.liang
"""
import itertools
from pickle_save import save, load
import spacy
import re

nlp=spacy.load("en")

#path="en_wiki_new_all_sent.pickle"
#sentences=list(word2vec.LineSentence(path))

"""
First load the pickled list of texts.
parameters:
   pickled_file: pickled list
    
"""

def all_text_sentences(texts_list,ngram,n_token):
    """
    parameter :
        texts_list
    """
    assert ngram >=1 
    #texts_list=load(pickled_file)
    sentences=[]
    for doc in nlp.pipe(texts_list,n_threads=4):
        n_token+=len(doc)
        if ngram==1:
            sentences.append(transform_sentences(doc))
        elif ngram >1:
            sentences.append(transform_sentences_ngram(doc,ngram))
    all_sentences=list(itertools.chain.from_iterable([sent for sent in sentences]))
    print("There are %d tokens."%n_token)
    return n_token, all_sentences
   # yield all_sentences
    
def transform_sentences(doc):
    """
    parameter: 
        text: string
    """    
    word=r"\w+"
    sentences=[]
    #doc = nlp(text.strip())
    
    for sent in doc.sents:
            a_sent=[]
            for token in sent:
                if re.search(word,token.text) != None and token.is_stop==False:
                   a_sent.append((token.text,token.tag_,token.dep_))
            sentences.append(a_sent) 
    return sentences
                   
    
def transform_sentences_ngram(doc,n): # bigram or more
    #doc = nlp(text.strip())
    word=r"\w+"
    sentences_ngram=[]
    for sent in doc.sents:
        a_sent=[]
        new_sent=[]
        for token in sent:
            if re.search(word,token.text) != None and token.is_stop==False:
                new_sent.append((token.text,token.tag_,token.dep_))
        while len(new_sent) <= n: 
            new_sent.append(("_","_","_"))
        for i in range(len(new_sent)-n):   
            span=new_sent[i:i+n]
            #print(span)
            words=[tok[0] for tok in span]
            tags=[tok[1] for tok in span]
            deps=[tok[2] for tok in span]
            a_sent.append(("_".join(words),"_".join(tags),"_".join(deps)))
        sentences_ngram.append(a_sent)   
    return sentences_ngram
        

def chunk(length, d=1000):
    sizes=[]
    ran=int(length/d)
    sizes=list(map(lambda x: [x*d,(x+1)*d], list(range(ran))))
    #for i in range(ran):
        #start=i*d
        #i_=i+1
        #end=i_*d
        #sizes.append([start,end])
    if length%d !=0:
        end=sizes[-1][1]
        sizes.append([end,end+length%d])
    
    return sizes  

 
"""  
path= "en_wiki_lastest_text_200.pickle" 
n_token=0
text_list=load(path)
length=len(text_list)
print(len(text_list),len(text_list[0]))
all_sentences_uni=[]

for size in chunk(length,d=1000):
    values_uni=[v for v in all_text_sentences(text_list[size[0]:size[1]],1,n_token)]
    all_sentences_uni+=values_uni[1]
    n_token+=values_uni[0]
    print("transformed %d sentences of unigram"%len(all_sentences_uni))
    print("got %d tokens"%n_token)
path_save="sentences_uni_list.pickle"
save(path_save,all_sentences_uni)
"""