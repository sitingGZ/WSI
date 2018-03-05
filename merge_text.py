#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:27:23 2018

@author: siting.liang
"""
import pickle
import spacy
import re
from gensim.models import word2vec
nlp=spacy.load("en")

path="en_wiki_201802_1.txt"
#sentences=list(word2vec.LineSentence(path))
word=r"\w+"
string ='The word "anarchism" is composed from the word "anarchy" and the suffix -ism, themselves derived respectively from the Greek , i.e. ''anarchy'' (from , ''anarchos'', meaning "one without rulers"; from the privative prefix ἀν- (''an-'', i.e. "without") and , ''archos'', i.e. "leader", "ruler"; (cf. ''archon'' or , ''arkhē'', i.e. "authority", "sovereignty", "realm", "magistracy")) and the suffix  or  (''-ismos'', ''-isma'', from the verbal infinitive suffix -ίζειν, ''-izein''). The first known use of this word was in 1539. Various factions within the French Revolution labelled opponents as anarchists (as Maximilien Robespierre did the Hébertists) although few shared many views of later anarchists. There would be many revolutionaries of the early nineteenth century who contributed to the anarchist doctrines of the next generation, such as William Godwin and Wilhelm Weitling, but they did not use the word "anarchist" or "anarchism" in describing themselves or their beliefs.'
#i_s=0
#i_t=0
#i_l=0
all_sent=[]
all_sent_token=[]
all_tags=[]
"""
for line in open(path,"r").readlines():
    i_l+=1
    doc =nlp(line.strip())
    for sent in doc.sents:
        i_s+=1
        print (sent)
        s=[]
        p=[]
        d=[]
        for token in sent:
            if re.search(word,token.text) != None:
               i_t+=1
               s.append(token.text)
               p.append(token.tag_+"_"+token.dep_)
               d.append((token.text,token.tag_,token.dep_))
           #print(token.text,token.pos_,token.tag_,token.is_stop)
        all_sent.append(d)
        all_sent_token.append(s)
        all_tags.append(p)
    
    
"""
print("there are %d tokens"%i_t)
print("there are %d sentences"%i_s)
print("there are %d texts"%i_l)
print(len(all_sent),len(all_sent_token),len(all_tags))
