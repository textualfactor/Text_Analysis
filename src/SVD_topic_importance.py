import os
import math
import gensim 
import SVD_TF
import numpy as np
import scipy.linalg as sci
import pandas as pd
import csv
import operator
import pickle


#choose the number of most important clusters that will be written in an csv
top_number = 500 
name = "top{}_important_clusters_google50_avg.csv".format(top_number)

'''
specify dictionary, dic_bow and load clusters
'''
dictionary_path = 'all_dictionary'
bow_path = 'all_bag_words.npy'
cluster_path ='./google_cluster50_processed.npy'
cluster = np.load(cluster_path)




'''
Run SVD for unseeded processed google clusters to get topic importance dictionary, {i:j, ...},
where i represents the i^th cluster in the cluster file and j is that cluster's importance derived from SVD.

Specically, after performing SVD on a cluster's document-term matrix, 
we calculate that cluster's importance by multiplying the l1 norm of the top left singular vector 
with the top singular value.
'''

error_list = []
topic_dict = {}
obj = SVD_TF.SVD_TF(dictionary=dictionary_path, bow=bow_path) #generate the svd object 

for i in range (0,len(cluster)):
    print("{} of {}".format(i, len(cluster)))
    #set document-term matrix associated with the cluster
    term_document_frequency_matrix = obj.set_term_document_matrix(cluster[i], tf_idf = False) 
    try:
        #perform SVD on the document-term matrix associated with the cluster
        t_ldings, sv, v_ldings = sci.svd(obj.get_term_document_matrix(), full_matrices = False) 
        accumulator_eqwgt = 0
        for loading in v_ldings[0]: #iterate over the top left singular vector
             #calculate the cluster's importance value
            accumulator_eqwgt = accumulator_eqwgt + (abs(loading) * abs(sv[0]))
        topic_dict[i] = accumulator_eqwgt    
    except:
        print("cluster {} skipped".format(i))
        #update the error list with the indices of the clusters that did not perform SVD
        error_list.append(i) 



'''
Save the clusters that performed SVD
''' 
clusters_used = np.delete(cluster, error_list)
np.save('google_cluster_50_used_all.npy', clusters_used) 



'''
Sort the topic importance dictionary by the clusters' importance value 
and save the sorted clusters to a txt file
'''
sorted_topic_dict = sorted(topic_dict.items(), key=operator.itemgetter(1), reverse= True)
cluster_ranked = []
for i in range(len(sorted_topic_dict)):
    cluster_ranked.append(cluster[sorted_topic_dict[i][0]])

with open("ranked_clusters_all.txt", "wb") as fp:
    pickle.dump(cluster_ranked, fp)


'''
Write the chosen number of most important clusters to a csv file
'''
with open(name, "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i in range(0, top_number):
        words = []
        for j in cluster_ranked[i]:
            words.append(j)
        words.insert(0, "Topic Importance{}".format(i))
        writer.writerow(words)

