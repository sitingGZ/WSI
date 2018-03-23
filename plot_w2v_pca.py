
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


from preprocess import transform_sent
import spacy
nlp=spacy.load("en_core_web_lg")

vec_dict={0:"tokens_uni.vec",1:"tag_dep_.vec",2:"tag_dep_head.vec",3:"cluster_uni.vec"}
bi_grams={0:"tokens_bi.vec",1:"tags_bi.vec"}


def get_vec(path):
    keyvectors=KeyedVectors.load_word2vec_format(path,binary=False)
    vocab=keyvectors.vocab
    vectors=[keyvectors.get_vector(w) for w in vocab]
    return vocab, vectors

def get_vocab_embed(w,keyvectors):
    
    #vocab=[v for v in keyvectors.vocab]
    #vectors=[]
    #for w in words:
    if w in keyvectors.vocab:
           vector=keyvectors.get_vector(w)
    else:
        vector=np.zeros(300)
    #print(len(vocab),type(vocab))
    return vector

def plot_words(words,embed_dict):
    #print(brown.sents()[:10])
    #while True:
        #pass
    #model = Word2Vec(brown.sents())
    #words=get_vocab(embed)[0]
    #embed_dict=get_vocab(embed)[1]
    
    #words = [word.strip() for word in open(path).readlines()]

    vectors = [embed_dict[word] for word in words if word in embed_dict]
    vectors = PCA(n_components=3).fit_transform(vectors) #andere Variante TSNE

    x, y = zip(*vectors)
    
    plt.quiver([0], [0], x, y, angles='xy', scale_units='xy', scale=0.1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()
    
def tsne_plot(labels,vectors):
    
    
    
    tsne_model=TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values=tsne_model.fit_transform(vectors)
    
    x=[]
    y=[]
    for value in new_values:
       
        x.append(value[0])
        y.append(value[1])
        
    print(value[0])
    print(value[1])
    
    plt.figure(figsize=(16,16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],xy=(x[i],y[i]),xytext=(5,2),textcoords="offset points", ha="right",va="bottom")
    plt.show()
    
    

   
    
def get_word_embed(new_sent,index):
    all_words=[]
    vectors=[]
    for i in index:

        keyvectors= KeyedVectors.load_word2vec_format(vec_dict[i],binary=False)
        for w in new_sent:
            all_words.append("_".join((w[0],w[i])))           
            vectors.append(get_vocab_embed(w[i],keyvectors))
            
        
        
    return all_words,vectors




def merge_word_embed(new_sent,index):
    labels=[]
    vectors=[]

    for w in new_sent:
        w_tpl=[]
        w_tpl.append(w[0])
        v_tpl=[]
        for i in index:
            w_tpl.append(w[i])
            v_tpl.append(get_vocab_embed(w[i],KeyedVectors.load_word2vec_format(vec_dict[i],binary=False)))
        
        labels.append("_".join(w_tpl))
        vectors.append(np.mean(np.array(v_tpl),axis=0))
    return labels,vectors
            
        
    
def make_doc_embed(sent,index):
    labels=[]
    vectors=[]
    new_sent=transform_sent(sent)
    words,vectors=merge_word_embed(new_sent,index)
            
    for w in new_sent:
        w_tpl=[]
        for i in index:
            w_tpl.append(w[i])
        labels.append("|".join(w_tpl))
    doc_vec=np.mean(np.array(vectors),axis=0)
    return labels,vectors,doc_vec
            
            
    #words=[w[index] for w in new_sent]
    #vector=np.asarray(get_vocab_embed(words,keyvectors))
    #return np.mean(vector,axis=0)
    

if __name__ == '__main__':
    
    sent1="I am going to look for a hotel."
    sent2="Please have a look at this hotel."
    sent3="Please look at this hotel."
    ##sent3=""
    sents=[sent1,sent2,sent3]
    #print(new_sents1)
    #print(new_sents2)
    
    index=[1,3]
    labels=get_vec("sense2vec.vec")[0]
    vectors=get_vec("sense2vec.vec")[1]
    #words=[v for v in keyvectors.vocab]
    #words=list(set([t[3] for sent in sents for t in transform_sent(sent)]))
    #labels=list(set(["|".join((t[0],t[3])) for sent in sents for t in transform_sent(sent)]))
    print(len(labels))
   
    #vectors=get_vocab_embed(words,keyvectors)
    
    
    print(len(vectors))
    #plot_words(words,embed_dict)
    #tsne_plot(labels,vectors)
    tsne_plot(labels,vectors)
    