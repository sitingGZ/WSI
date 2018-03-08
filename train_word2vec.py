#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:19:39 2018

@author: siting.liang
"""

from pickle_save import load
import gensim.models

all_sents_part=load("sentences_uni_list.pickle")
model_uni_part=gensim.models.Word2Vec(all_sents_part,iter=1)