#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:12:20 2018

@author: siting.liang
"""
from sentence_generator import *
import logging
import gensim.models
import sys
import argparse
import multiprocessing as mp

texts="text_200.txt"

#example: python training.py sents_token sents_token.vec -s 300 -w 5 -mc 5  
parser= argparse.ArgumentParser(description='Script for training word vector models using enwiki corpora of 20180220')
parser.add_argument('corp_type', type=str, help='preprocessed corpora tpye:tokens_uni,tokens_bi,tags_deps_uni,tags_dep_heads_uni,sense2vec')
parser.add_argument('target', type=str, help='target file name to store model in')
parser.add_argument('-s', '--size', type=int, default=300, help='dimension of word vectors')
parser.add_argument('-w', '--window', type=int, default=5, help='size of the sliding window')
parser.add_argument('-mc', '--mincount', type=int, default=5, help='minimum number of occurences of a word to be considered')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='number of worker threads to train the model')
parser.add_argument('-g', '--sg', type=int, default=1, help='training algorithm: Skip-Gram (1), otherwise CBOW (0)')
parser.add_argument('-i', '--hs', type=int, default=1, help='use of hierachical sampling for training')
parser.add_argument('-n', '--negative', type=int, default=0, help='use of negative sampling for training (usually between 5-20)')
parser.add_argument('-o', '--cbowmean', type=int, default=0, help='for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors')
args = parser.parse_args()
logging.basicConfig(
    filename=args.target.strip() + '.result', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)

corporas={"tokens_uni":sents_token(texts,1),"cluster_uni":sents_cluster(texts),"tokens_bi":sents_token(texts,2),"tags_deps_uni":sents_ngram_tags(texts,1),"tags_bi":sents_ngram_tags(texts,2),"sents_tags_dep_heads_uni":sents_tags_heads(texts),"sense2vec":sense2vec(texts)}

#get sentences generator
sentences=corporas[args.corp_type]

# train the model
model = gensim.models.FastText(size=args.size,window=args.window,min_count=args.mincount,workers=args.threads,sg=args.sg,hs=args.hs,negative=args.negative,word_ngrams=0)
model.build_vocab(sentences)
model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)

# store model


model.save(args.target)
