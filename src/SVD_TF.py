#Importing Requisite Packages
import numpy as np
import gensim
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import os
import math
import scipy.linalg as sci
import timeit

class SVD_TF:
#Performs Single Value Decomposition on term-doc freq matrix which can be simple
#or have  optional tfidf encoding

    def __init__(self, dictionary, bow):
            #params --
            #dictionary: keys: word id, values: word
            # bow: dim(numdocs X num_words X 2) array of word_id, word_freq pairs
            # bow[0] = [(0,1), (9, 8), ..., (2,3)],
            #   where word with id=0 occurs in doc1 1 time, word with id=9 occurs 8 times
            # bow[1] = [(3,1), (2, 6), ..., (12,3)]
            #   where word with id=3 occurs in doc2 1 time
            if isinstance(dictionary, str): #RECCOMENDED Takes in path of dictionary
                self.dictionary = gensim.utils.SaveLoad.load(dictionary)
            else:
                self.dictionary = dictionary
            if isinstance(bow, str): #RECOMMEDNED Takes in path of bow
                self.bow = np.load(bow, allow_pickle=True)
            else:
                self.bow = bow
            corpus_dict = []
            for i in range (0, len(self.bow)):
                corpus_dict.append({key: value for key, value in self.bow[i]})
            self.corpus_dict = corpus_dict

    def __filter_cleaner(self, topic):
        # Function that cleans word comprising topic of symbols
        # params --
        # topic: array of words used to determine term_doc_matrix
        topic_cleaned = []
        for word in topic:
            word = word.replace('*','')
            word = word.replace('-','')
            word = word.replace("'",'')
            word = word.replace(",",'')
            word = word.replace("!",'')
            word = word.replace("@",'')
            word = word.replace("#",'')
            word = word.replace("$",'')
            word = word.replace("%",'')
            word = word.replace("&",'')
            word = word.replace("(",'')
            word = word.replace(")",'')
            if (word != ''):
                topic_cleaned.append(word)
        return topic_cleaned

    def SVD(self, topic, tf_idf = False):
        #Function performs SVD decomposition
        #Sets resultinig word loading (topic vec )and document loading for topic
        # params --
        # topic : array of words (strings)
        # tf_idf : specifies encoding of doc_term_mat
        # 	False => simple word_freq in doc count
        # 	True => tf_idf encoding
        self.topic = self.__filter_cleaner(topic) #Topic gets filtered for symbols
        self.set_term_document_matrix(self.topic, tf_idf)
        self.t_ldings, self.sv, self.v_ldings = sci.svd(self.get_term_document_matrix(), full_matrices = False)
    def set_term_document_matrix(self, topic, tf_idf):
        # Set document_term_freq_matrix associated with a topic
        # m - number of words in topic
        # n - number of documents in corpus
        # matrix ~ R^(m * n) matrix w_(i,j) is count of word i in document j
        # if tf_idf == True, w_(i,j) is tf_idf encoding of word i in document j
        # Note: for computation easiness, we generate the document-term-matrix as the transpose of the one
        # defined in Algorithm 4 in the paper; 
        # taking the transpose will not affect SVD except that left singluar vectors become right singular vectors and right singular vectors become left.


        #params --
        #topic: array of words
        # tf_idf : specifies encoding of doc_term_mat


        topic_ids = self.__get_ids(topic)
        #Getting Dimensions of Matrix
        self.n = len(self.bow) #n is number of documents
        self.m = len(topic_ids) #m is number of words (size of vocabulary)

        #Instantiating Term-Document Frequency Matrix
        matrix = np.zeros(shape = (self.m,self.n), dtype= np.int64)
            #filling_matrix
        if (tf_idf == True):
            for m_i in range (0,self.m):
                d = self.__occurances_in_corpus(topic_ids[m_i])
                matrix[m_i] = self.__word_documents_freq(topic_ids[m_i]) * math.log (n/d)
        else:
            for m_i in range (0,self.m):
                matrix[m_i] = self.__word_documents_freq(topic_ids[m_i])
        self.term_document_frequency_matrix = matrix
    
    def get_term_document_matrix(self):
        return self.term_document_frequency_matrix
    
    def __word_documents_freq(self,word_id):
        # Given a word, get the number of its occurence in each documnet 
        word_counts_over_doc = np.zeros(shape = (1,self.n), dtype= np.int64) # initialize a 1 x n array n - number of documents
        for doc_num in range (0, self.n):
                try: word_counts_over_doc[0][doc_num] = self.corpus_dict[doc_num][word_id]
                except KeyError:
                    continue
        return word_counts_over_doc
    
    def __occurances_in_corpus(self, word_id):
        # Given a words, get the number of documents it is in
        not_in_documents = 0
        for document_number in range(0,len(bow)):
            try: corpus_dict[document_number][word_id]
            except KeyError:
                not_in_documents +=1
        return len(bow) - not_in_documents
    def __get_ids(self, topic):
        # Get the id of the words in the cluster from Gensim dictionary (generated in step3)
        topic_ids = []
        for word in topic:
            if self.dictionary.doc2idx([word])[0] == -1:
                continue
            topic_ids.append(self.dictionary.doc2idx([word])[0])
        return np.asarray(topic_ids)
    def topic_loading(self):
        # get the top left singular vector (top right singular vector of the matrix defined in the paper)
        return  self.t_ldings[:,0]
    def document_loadings(self):
        # get the top right singular vector (top left singular vector of the matrix defined in the paper)
        return self.v_ldings[0]


#############################################################################
#------------------------------GRAPHING SECTION----------------------------#
    def graph_topic_over_documents(self, doc_init = 0, doc_fin = 0, step = 'auto', title = 0):
        #Graphing Document Loadings on Topic
        #params --
        # doc_init (int): the first doc the user wants to graph, allowing user to take
        #partition of corpus
        # doc_fin (int): the last doc the user wants to graph. The function will Then
        # graph all documents from doc_init to doc_fin
        # step (str or int): 'auto' lets pyplot determine best x-tick step
        # if step is set to number, x-ticks are determined manually
        # save (str): if =='save', the function will save the graph to the current
        # working directory. else function will not save graph
        #title : the name the user wants to give the graph

        if (doc_fin == 0):
            doc_fin = len(self.get_term_document_matrix())
        plt.figure(1)
        if isinstance(step, int):
            plt.xticks(np.arange(0, (doc_fin-doc_init), step), [doc_id for doc_id in np.arange(doc_init, doc_fin, step)])


        #Labelling Axis and Titling
        if (title == 0):
            title = self.topic[0]
        plt.title("{}: Topic vs. Documents".format(title.title()))
        plt.ylabel("Loadings")
        plt.xlabel('Document Number')

        #Plotting Figure
        plt.plot(abs(self.document_loadings()[doc_init:doc_fin]))

        return plt.figure(1)
    def save_graph(self, graph, PATH = "", name = ""):
        #Function saves given graph
        #params --
        #graph: the plt.figure(1) graph object
        #PATH: The location for the graph:
        #name: the file name

        #Saving Figure
        if not os.path.exists("{}".format(name)):
            os.makedirs("{}".format(name))
        graph.savefig("{}/{}-SVD.png".format(name, name.title()), dpi = 1000,  bbox_inches='tight')
