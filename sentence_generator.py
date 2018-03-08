#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:43:39 2018

@author: siting.liang
"""
from pickle_save import load
from text2sents import transform_sentences,transform_sentences_ngram
import spacy
nlp=spacy.load("en")
import gensim.models

"""
Objects are different sentences types which are going to be used for word embedding training.
sents_token : takes word token as vocabulary, can be unigramm, bigramm, or trigramm
sents_tags: takes pos tag and dependency as vocabulary, can be unigramm, bigramm, or trigramm
parameters : 
      texts_list:
           list of all texts which are extracted from enwiki corpus.
      ngram:
           int, define to make ngram model. 
    
"""

class sents_token(object):
    def __init__(self,texts_list):
        self.texts_list=texts_list
        
    def __iter__(self): 
        for text in self.texts_list:
            doc=nlp(text)
            sentences=transform_sentences(doc)
            yield [[token[0] for token in sent] for sent in sentences]
            yield [["_".join(token[1:]) for token in sent] for sent in sentences]
            
class sents_ngram_token(object):
    def __init__(self,texts_list,ngram):
        assert ngram >=1 
        self.texts_list=texts_list
        self.ngram=ngram
    
    def __iter__(self):
        for text in self.texts_list:
            doc=nlp(text)
            if self.ngram==1:
                sentences=transform_sentences(doc)
            elif self.ngram >1:
                sentences=transform_sentences_ngram(doc, self.ngram)
            for sent in sentences:                
                yield [token[0] for token in sent ]
            #yield [["_".join(token[1:]) for token in sent] for sent in sentences] 
            
class sents_ngram_tags(object):
    def __init__(self,texts_list,ngram):
        assert ngram >=1 
        self.texts_list=texts_list
        self.ngram=ngram
    
    def __iter__(self):
        for text in self.texts_list:
            doc=nlp(text)
            if self.ngram==1:
                sentences=transform_sentences(doc)
            elif self.ngram >1:
                sentences=transform_sentences_ngram(doc, self.ngram)
            #yield [[token[0] for token in sent] for sent in sentences]
            for sent in sentences:
                
                 yield ["_".join(token) for token in sent ] 

#all_sents_200=load("en_wiki_lastest_text_200.pickle")
#all_sents_100=load("en_wiki_lastest_text_100.pickle")
#sents_uni_token=sents_ngram_token(all_sents_100,1)
#sents_bi_token=sents_ngram_token(all_sents_200,2)
#sents_uni_tags=sents_ngram_tags(all_sents_200,1)
#sents_bi_tags=sents_ngram_tags(all_sents_200,2)


