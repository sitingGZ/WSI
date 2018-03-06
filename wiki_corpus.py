#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:09:06 2018

@author: siting.liang
"""

from gensim.corpora.wikicorpus import WikiCorpus
import re
from spacy.lang.en import English
nlp=English()

tokenizer = English().Defaults.create_tokenizer(nlp)

path="enwiki-20180220-pages-meta-current1.xml-p10p30303.bz2"



wiki=WikiCorpus(path,processes=None, lemmatize=False,dictionary={},token_min_len=2, lower=False)
sentences=list(wiki.get_texts())



def write_file(s_list,path):
    file=open("current"+path+".txt","w")
    text=" "
    for s in s_list:
        text=" ".join(s)
        
        file.write(text+"\n")
    print (text)

m=re.search("current(.*)xml",path)        
save_to=m.group(1)
write_file(sentences, save_to)