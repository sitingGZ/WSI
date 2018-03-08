#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:12:20 2018

@author: siting.liang
"""
from sentence_generator import *
import gensim
import gensim.models

model_uni_words=gensim.models.Word2Vec(sents_uni_token,iter=1)