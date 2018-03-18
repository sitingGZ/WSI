#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:27:23 2018

@author: siting.liang
"""
import itertools

import re

#nlp=spacy.load("en_core_web_lg")

#path="en_wiki_new_all_sent.pickle"
#sentences=list(word2vec.LineSentence(path))

"""
First load the pickled list of texts.
parameters:
   pickled_file: pickled list
    
"""
"""
def all_text_sentences(texts_list,ngram,n_token):  
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
"""
def ent_merge(doc):
    """
    merge the name entities over the doc. 
    """
    for ent in doc.ents:
        ent.merge(tag=ent.root.tag_,lemma=ent.lemma_,ent_type=ent.label_)   
    return doc

def transform_sentences(doc):
    """
    parameter: 
        text: spacy nlp(string)
    return:
        a list of sentences from a spacy doc, form of a sentence with each word as a tuple with: 
        (text,cluster number, tag, dependency)
        
    """    
    
    #doc = nlp(text.strip())
    word=r"\w+"
    sentences=[]

    for sent in ent_merge(doc).sents:
            a_sent=[]
            for token in sent:
                #filter the puntuations, numbers, and stop words. 
                if re.search(word,token.text) != None and token.is_digit ==False and token.is_stop==False:
                   a_sent.append((token.text,token.tag_,token.head.tag_,token.dep_,token.head.dep_))
            sentences.append(a_sent) 
    return sentences

def transform_brown_cluster(doc):
    """
    parameter: 
        text: spacy nlp(string)
    return:
        a list of sentences from a spacy doc, in which each word is represented by its
        brown cluster number.
    """    
    
    #doc = nlp(text.strip())
   
    sentences=[]

    for sent in doc.sents:
            a_sent=[]
            for token in sent:
                #filter the puntuations, numbers, and stop words. 
                   a_sent.append(str(token.cluster))
            sentences.append(a_sent) 
    return sentences
                                     
    
def transform_sentences_ngram(doc,n): # bigram or more
    #doc = nlp(text.strip())
    """
    return:
        a list of sentences , each sentence has the form as the bigramm example : every bigramm word is a tuple contains (text,cluster number, tag, dependency)
        [('The_word', '30_6853', 'DT_NN', 'det_nsubjpass'), ('word_anarchism', '6853_0', 'NN_NN', 'nsubjpass_appos'), ('anarchism_is', '0_762', 'NN_VBZ', 'appos_auxpass'), ('is_composed', '762_2005', 'VBZ_VBN', 'auxpass_ROOT'), ('composed_from', '2005_380', 'VBN_IN', 'ROOT_prep'), ('from_the', '380_11', 'IN_DT', 'prep_det'), ('the_word', '11_6853', 'DT_NN', 'det_pobj'), ('word_anarchy', '6853_1893', 'NN_NN', 'pobj_appos'), ('anarchy_and', '1893_20', 'NN_CC', 'appos_cc'), ('and_the', '20_11', 'CC_DT', 'cc_det'), ('the_suffix', '11_6853', 'DT_NN', 'det_conj'), ('suffix_-ism', '6853_0', 'NN_NN', 'conj_appos'), ('-ism_themselves', '0_8042', 'NN_PRP', 'appos_nsubj'), ('themselves_derived', '8042_16170', 'PRP_VBD', 'nsubj_acl'), ('derived_respectively', '16170_26', 'VBD_RB', 'acl_advmod'), ('respectively_from', '26_380', 'RB_IN', 'advmod_prep'), ('from_the', '380_11', 'IN_DT', 'prep_det'), ('the_Greek', '11_151', 'DT_NNP', 'det_pobj'), ('Greek_i.e.', '151_2004', 'NNP_FW', 'pobj_advmod'), ('i.e._anarchy', '2004_1893', 'FW_NN', 'advmod_appos'), ('anarchy_from', '1893_380', 'NN_IN', 'appos_prep'), ('from_anarchos', '380_0', 'IN_NN', 'prep_conj'), ('anarchos_meaning', '0_31146', 'NN_VBG', 'conj_advcl'), ('meaning_one', '31146_8170', 'VBG_CD', 'advcl_dobj'), ('one_without', '8170_57340', 'CD_IN', 'dobj_prep'), ('without_rulers', '57340_4077', 'IN_NNS', 'prep_pobj'), ('rulers_from', '4077_380', 'NNS_IN', 'pobj_prep'), ('from_the', '380_11', 'IN_DT', 'prep_det'), ('the_privative', '11_0', 'DT_JJ', 'det_amod'), ('privative_prefix', '0_2245', 'JJ_NN', 'amod_pobj')]
    """
    word=r"\w+"
    sentences_ngram=[]
    
    for sent in doc.sents:
        a_sent=[]
        
        new_sent=[]
        for token in sent:
            if re.search(word,token.text) != None and token.is_stop==False:
                new_sent.append((token.text,token.tag_))
        while len(new_sent) <= n: 
            new_sent.append(("_","_"))
        for i in range(len(new_sent)-n):   
            span=new_sent[i:i+n]
            #print(span)
            words=[tok[0] for tok in span]
            tags=[tok[1] for tok in span]
            a_sent.append(("_".join(words),"_".join(tags)))
            
        sentences_ngram.append(a_sent) 
        
    return sentences_ngram


        
def sense2vec_transform(doc):
    sentences=[]
    for ent in doc.ents:
                ent.merge(tag=ent.root.tag_,lemma=ent.text,ent_type=ent.label_)

    for np in doc.noun_chunks:
            while len(np)>1 and np[0].dep_ not in ("advmod","amod","compound"):
                  np=np[1:]

            np.merge(tag=np.root.tag_,lemma=np.text,ent_type=np.root.ent_type_)
    for sent in doc.sents:   
        tokens=[]
        for token in sent:
            text=token.text.replace(' ','_')
            tag=token.ent_type_ or token.pos_
            tokens.append('{}|{}'.format(text,tag))
        sentences.append(tokens)
    return sentences

"""    
text="The Winter War (30 November 1939 – 13 March 1940) began when the Soviet Union (USSR) invaded Finland three months after the outbreak of World War II. The USSR had sought to annex Finnish territory, including land near Leningrad, 32 km (20 mi) from the border. After Finland refused, the USSR attacked with more than twice as many soldiers, thirty times as many aircraft, and a hundred times as many tanks as the defending forces. The Red Army had been crippled by Joseph Stalin's Great Purge and the Finnish Defence Forces repelled the invasion in temperatures down to −43 °C (−45 °F) for much longer than expected. A reorganized Soviet offensive broke through in February 1940 and forced the Finns to seek peace. Finland ceded 11 percent of its territory, but retained sovereignty. Soviet casualties have been estimated at 321,000 to 381,000, compared to Finnish casualties of 70,000. The poor performance of the Red Army encouraged Adolf Hitler to consider an attack on the USSR. After a 15-month lull called the Interim Peace, the Continuation War and Operation Barbarossa began in June 1941. "


doc=nlp(text)
sentences=sense2vec_transform(doc)
print(len(sentences),sentences)


Testing:

path= "/home/students/liang/Documents/2017WiSe/FS/Project/enwiki/en_wiki_lastest_text_200.pickle" 
n_token=0
text_list=load(path)
length=len(text_list)
print(len(text_list),len(text_list[0]))
all_sentences_uni=[]
doc=nlp(text_list[0])
print(transform_sentences_ngram(doc,2)[1])

 
for size in chunk(length,d=1000):
    values_uni=[v for v in all_text_sentences(text_list[size[0]:size[1]],1,n_token)]
    all_sentences_uni+=values_uni[1]
    n_token+=values_uni[0]
    print("transformed %d sentences of unigram"%len(all_sentences_uni))
    print("got %d tokens"%n_token)
path_save="sentences_uni_list.pickle"
save(path_save,all_sentences_uni)
"""
