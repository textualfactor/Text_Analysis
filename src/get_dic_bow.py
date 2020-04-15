'''
Create dictionary and bag of words (bow) from tokenized textual data, which are 
required as inputs for SVD.

dictionary (gensim object): keys: word id, values: word
bag of words (bow): dim(num_of_docs * num__of)words X 2) array of word_id, word_freq pairs
bow[0] = [(0,1), (9, 8), ..., (2,3)], where word with id=0 occurs in doc1 1 time, word with id=9 occurs 8 times
bow[1] = [(3,1), (2, 6), ..., (12,3)], where word with id=3 occurs in doc2 1 time, ...
'''
import numpy as np
import gensim
import pandas as pd
import os
import math

#used for testing only
import timeit

from collections import defaultdict
from gensim.corpora import Dictionary


def df_to_dict(df):
    # Get defaultdict (Python dictionary) indexing lists of (word,count) pairs by document
    return_dict = defaultdict(list)
    for (_,doc,word,count) in df.itertuples():
        return_dict[doc].append((word,count))
    return return_dict


def create_dictionary(doc_dict):
    # create gensim dictionary by using python dictionary as input
    dic = Dictionary()
    for doc in doc_dict:
        dic.add_documents([[word for (word,_) in doc_dict[doc]]])
    dic.save(dic_path)
    return dic


def create_bag_of_words(doc_dict,dic):
    # create bag of words
    bow_list = []
    for doc in doc_dict:
        bow_list.append([[dic.doc2idx([word])[0],count] for (word,count) in doc_dict[doc]])
    arr = np.array(bow_list)
    np.save(npy_path,arr)
    return arr


def dic_bow():
    colnames = ['document', 'word', 'counts']
    df = pd.read_csv(csv_path, names=colnames)
    doc_dict = df_to_dict(df)
    dic = create_dictionary(doc_dict)
    create_bag_of_words(doc_dict,dic)

    #timers for individual functions
    #n = 10
    #print('df_to_dict ({} trials):'.format(n),timeit.timeit(lambda : df_to_dict(df),number = n))
    #print('create_dictionary ({} trials):'.format(n),timeit.timeit(lambda : create_dictionary(doc_dict),number = n))
    #print('create_bag_of_words ({} trials):'.format(n),timeit.timeit(lambda : create_bag_of_words(doc_dict,dic),number = n))

#if __name__ == '__main__':
#    #naming schemes for csv, dic, and npy files
#    title = 'all'
#    csv_path = 'output/output_' + title + '.csv'
#    dic_path = title + '_dictionary'
#    npy_path = title + '_bag_words.npy'
#    main()
#    
    #timer for entire program
    #print(timeit.timeit(main,number=10))