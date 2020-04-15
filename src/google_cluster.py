'''
Generate clusters from google trained vectors given specified size 
and cluster generating algorithm (hierarchical by default)
'''
import Cluster_TF
import numpy as np
model_path = './GoogleNews-vectors-negative300.bin'

def google_cluster(cluster_size=50):
    #Create Cluster_TF object and use LSH to generate clusters
    obj = Cluster_TF.Cluster_TF(vector_file=model_path,  number_of_words= 100000)
    cluster_all = obj.cluster_chained(m=cluster_size, hierarchical = 1,
    number_of_queries = 1000, query_accuracy = .9, number_of_tables = 160, hash_bit = 18)
    # save the generated clusters to txt file and numpy file
    textfile = open('google_cluster_{}.txt'.format(cluster_size),'w')
    for cluster in cluster_all:
        textfile.write(str(cluster))
        textfile.write('\n')
    textfile.close()
    cluster_npy = np.array(cluster_all)
    np.save('google_cluster_{}.npy'.format(cluster_size), cluster_npy)




