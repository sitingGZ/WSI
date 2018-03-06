#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:29:12 2018

@author: siting.liang
"""

from gensim.utils import smart_open
from pickle_save import save
import json
"""
First convert the wikipedia dump bz2 file into json.gz file using python -m gensim.scripts.segment_wiki,
Extract section texts from articles. 
parameters:
    pickle_path: string, path to save the result list.
    json_gz: string, json.gz file which contains wiki articles
    token_n: int , only extract texts which have more than exact number of tokens.
"""

def extract_text(pickle_path,json_gz,token_n):
    i=0
    text_list=[]
    for line in smart_open(json_gz):
        article=json.loads(line)
        i=+1
        for section_text in article['section_texts']:
            text=section_text.split("\n")
            for string in text:
                s=string.split(" ")
                if len (s) > token_n:
                   text_list.append(" ".join(s)+"\n")
                   
    # save text_list to binary file, can be loaded with load function from pickle_save         
    save(pickle_path,text_list)
    print("Saved %d text from %d articles."%(len(text_list),i))
    