#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:43:39 2018

@author: siting.liang
"""

from text2sents import *
import spacy
nlp=spacy.load("en_core_web_lg")




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
"""


class sense2vec(object): # only unigramm
    def __init__(self,texts):
        
        self.texts=texts
        
    
    def __iter__(self):
        
       with open(self.texts,"r") as f:
           for text in f:
               doc=nlp(text.strip())
           
               sentences=sense2vec_transform(doc)
            
               for sent in sentences:                
                   yield sent
                
class sents_token(object):
    def __init__(self,texts,ngram):
        assert ngram >=1
        self.texts=texts 
        self.ngram=ngram
    def __iter__(self):
        
        with open(self.texts,"r") as f:
           for text in f:
               doc=nlp(text.strip())
               if self.ngram ==1:
                  sentences=transform_sentences(doc)
                  for sent in sentences:                
                     yield [w[0] for w in sent]
               else:# transform_sentences_ngram returns a tuple of list , one without heads tags, the second one with heads 
                    sentences=transform_sentences_ngram(doc,self.ngram) 
                    for sent in sentences:                
                        yield [w[0] for w in sent]
             
            #['Various', 'factions', 'within', 'the French Revolution', 'labelled', 'opponents', 'as', 'anarchists', 'as', 'Maximilien Robespierre', 'did', 'the', 'Hébertists', 'although', 'few', 'shared', 'many', 'views', 'of', 'later', 'anarchists']

           

            
class sents_cluster(object): # only unigramm
    def __init__(self,texts):
        
        self.texts=texts
        
    
    def __iter__(self):
        with open(self.texts,"r") as f:
           for text in f:
               doc=nlp(text.strip())
           
               sentences=transform_brown_cluster(doc)
            
               for sent in sentences:                
                   yield sent
           
            
class sents_ngram_tags(object):
    # if n =1 , then ['NNP_nsubj', 'VBD_ROOT', 'DT_det', 'NN_attr', 'WDT_nsubj', 'NN_advmod', 'VBD_relcl', 'NNP_dobj', 'IN_prep', 'NNP_pobj', 'RBR_advmod', 'RB_advmod', 'IN_prep', 'NNP_pobj', 'IN_prep']
    # if n >1 ,then 
    def __init__(self,texts,ngram):
        assert ngram >=1 
        self.texts=texts
        self.ngram=ngram
    
    def __iter__(self):
       with open(self.texts,"r") as f:
           for text in f:
               sentences=[]
               doc=nlp(text.strip())
               if self.ngram==1: # train tag_dep
                  sentences=transform_sentences(doc)
                  for sent in sentences:
                      yield ["_".join((token[1],token[3])) for token in sent]
               elif self.ngram >1:
                   sentences=transform_sentences_ngram(doc, self.ngram)
            
                   for sent in sentences:
                
                        yield [token[1] for token in sent ] 
               

class sents_tags_heads(object):
      def __init__(self,texts):
    
        self.texts=texts
    
      def __iter__(self):
        with open(self.texts,"r") as f:
           for text in f:
               sentences=[]
               doc=nlp(text.strip())
               sentences=transform_sentences(doc)
               for sent in sentences:
                
                   yield ["_".join(token[2:]) for token in sent ]     
        #['DT_NN_det_nsubjpass', 'NN_VBN_nsubjpass_ROOT', 'NN_NN_appos_nsubjpass', 'VBZ_VBN_auxpass_ROOT', 'VBN_VBN_ROOT_ROOT', 'IN_VBN_prep_ROOT', 'DT_NN_det_pobj', 'NN_IN_pobj_prep', 'NN_NN_appos_pobj', 'CC_NN_cc_pobj', 'DT_NN_det_conj', 'NN_NN_conj_pobj', 'NN_NN_appos_conj', 'PRP_VBD_nsubj_acl', 'VBD_NN_acl_conj', 'RB_VBD_advmod_acl', 'IN_VBD_prep_acl', 'DT_NNP_det_pobj', 'NNP_IN_pobj_prep', 'FW_NN_advmod_appos', 'NN_NNP_appos_pobj', 'IN_NN_prep_conj', 'NN_IN_pobj_prep', 'VBG_NN_acl_conj', 'CD_VBG_dobj_acl', 'IN_CD_prep_dobj', 'NNS_IN_pobj_prep', 'IN_VBN_prep_ROOT', 'DT_NN_det_pobj', 'JJ_NN_amod_pobj', 'NN_IN_pobj_prep', 'XX_VBN_punct_ROOT']


