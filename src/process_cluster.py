import pandas as pd
import numpy as np
from nltk.corpus import stopwords


'''
Load relevant inputs
'''
# load vocabulary from the corpus: the tokenized textual data is stored in a csv file
# with first column containing document name, second column containing words in that document and thrid column containing word frequency
colnames = ['document', 'word', 'counts']
df = pd.read_csv('output/output_all.csv', names=colnames)
words = df.word.tolist()
words = list(set(words))
# load google unseeded clusters: the numpy file generated in step1
clusters = np.load('google_cluster_50.npy')
clusters_all = clusters.tolist()
# load support words from nltk package
stop_words = set(stopwords.words('english'))



def overlap(clusters, words):
	'''
	input: clusters: a list of clusters (list of words); 
	        words: a list of words from the corpus
	output: clusters: the intersection of the input clusters and words in corpus
	'''
	
	for cluster in clusters:
		for i in cluster:
			if i not in words:
				cluster.remove(i)
	return clusters


def lower(clusters):
	'''
	input: clusters: a list of clusters (list of words); 
	output: cluster_lower: the input clusters in lower case
	'''

	cluster_lower = []
	for cluster_0 in clusters:
		cluster = []
		for word in cluster_0:
			lower_word = word.lower()
			cluster.append(lower_word)
		cluster_lower.append(cluster)
	return(cluster_lower)


def repeated(clusters):
	'''
	input: clusters: a list of clusters (list of words); 
	output: cluster_rep: the input clusters without repeated words
	'''

	cluster_rep = []
	for cluster in clusters:
		cluster_new = list(set(cluster))
		cluster_rep.append(cluster_new)
	return cluster_rep

def stop(clusters, stop_words):
	'''
	input: clusters: a list of clusters (list of words); 
		   stop_words: a list of stop words from nltk package
	output: cluster_rep: the input clusters without stop words
	'''

	cluster_stop = []
	for cluster in clusters:
		cluster_new = [w for w in cluster if not w in stop_words] 
		cluster_stop.append(cluster_new)


'''
Process the google clusters by first intersecting them with words in textual data, 
then removing repeated words, and finally removing stop words
'''


def cluster_process():
    cluster_overlap = overlap(clusters_all, words)
    cluster_overlap_lower = lower(cluster_overlap)
    cluster_overlaprep = repeated(cluster_overlap_lower)
    clusters_overlaprep_stop = stop(cluster_overlaprep, stop_words)
    '''
    Save the processed clusters in a numpy file and text file
    '''
    cluster_npy = np.array(clusters_overlaprep_stop)
    np.save('google_cluster50_processed.npy', cluster_npy)
    textfile = open('google_cluster50_processed.txt','w')
    for cluster in clusters_overlaprep_stop:
        textfile.write(str(cluster))
        textfile.write('\n')
    textfile.close()

