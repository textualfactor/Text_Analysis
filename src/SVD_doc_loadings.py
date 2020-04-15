import SVD_TF
import numpy as np
import scipy.linalg as sci
import pandas as pd
import timeit
'''
Load dictionary, bow and clusters to perform SVD on.
'''
cluster_path ='./mdna_google_cluster50_processed.npy'
dictionary_path = 'all_dictionary'
bow_path = 'all_bag_words.npy'
clusters = np.load(cluster_path, allow_pickle=True)


'''
Load tokenized textual data to get the document names in the corpus while preserving order
'''
def uniq(seq):
	'''
	Get the unique elements in a list while preserving the original order
	'''
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]



def SVD_doc_load():
    colnames = ['document', 'word', 'counts']
    df = pd.read_csv('output/output_all.csv', names=colnames)
    docs = df.document.tolist()
    docs = uniq(docs)
    print("csv loaded and document names extracted")
    
    '''
    Perform SVD on each cluster and get their document loadings, doc_ldings. 
    doc_ldings is a numpy array of shape m * n with m being the number of clusters and n being the number of documents
    A cluster's document loading is the top left singular vector scaled by the top singular value
    '''
    doc_ldings = np.zeros((1, len(docs))) 
    error_list = []
    start = timeit.default_timer()
    obj = SVD_TF.SVD_TF(dictionary=dictionary_path, bow=bow_path) #generate the svd object 
    
    for i in range(len(clusters)):
        #print("{} of {}".format(i, len(clusters)))
        #set document-term matrix associated with the cluster
        term_document_frequency_matrix = obj.set_term_document_matrix(clusters[i], tf_idf = False) 
        try: 
            #perform SVD on the document-term matrix associated with the cluster
            t_ldings, sv, v_ldings = sci.svd(obj.get_term_document_matrix(), full_matrices = False) 
            #calculate the cluster's document loading
            document_loading = abs(v_ldings[0]) * abs(sv[0])
            doc_ldings = np.append(doc_ldings, np.array([document_loading]), axis=0)
            print("doc loadings added")
            print("{} of {}".format(i, len(clusters)))
        except: 
            print("cluster {} skipped".format(i))
            #update the error list with the indices of the clusters that did not perform SVD
            error_list.append(i) 
    end = timeit.default_timer()
    seconds = end - start
    minutes = round(seconds/60, 2)
    print('performing SVD of clusters of size 50: {} seconds; {} minutes'.format(seconds, minutes))
    doc_ldings_np = np.delete(doc_ldings, 0, 0)
    
    '''
    Save the clusters that performed SVD
    ''' 
    clusters_used = np.delete(clusters, error_list)
    np.save('mdna_all_google_50_used.npy', clusters_used)
    
    '''
    Write clusters' document loadings to a csv file, 
    with column headers being document name and row header being the first word in the cluster
    '''
    row_label = []
    for cluster in clusters_used:
        row_label.append(cluster[0])
    df = pd.DataFrame(doc_ldings_np, columns = docs, index=row_label)
    df.to_csv('mdna_all_50document_loadings_google.csv')






    




