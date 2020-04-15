#Importing requisite packages
import numpy as np
import falconn
import timeit
import math
import random
import matplotlib.pyplot
import gensim

class Cluster_TF:
#Class Description
# 1) Vectorizes user-given array of words
# by matching words with corresponding (key match)
# vectors in google/glove
# 2) Groups similar vectors together (~ related words)
# with both vanilla and sophisticated variations of hierarchical CLUSTERING
#   2.b) Uses locality sensitive hashing to partition vectors
#       in order to make nearest neighbor search possible
#3) Returns the clustered keys (actual word), which loosely correspond to topic
#EXAMPLE: data_set = [apple, banana, car, exchange, truck]
#         Clusters(data_set, numClusters = 3)
#         >> [[apple, banna], [car, truck], exchange]

    def __init__(self, vector_file, data_set = 0, number_of_words = 100000):
        # Constructor
        #params ---
        # data_set: 1D numpy array of strings (ie words). These words will be vectorized using google w2v or Glove Vectors (Stanford)
        # vector_file: This file is either GoogleNews-vectors-negative300.bin or glove .txt file
        # number_of_words : Specifies the size of the subset of the GoogleNews-vectors-negative300.bin that you want to compare to your dataset
        #...Using a subset of GoogleNews-vectors-negative300.bin is much faster than the entire GoogleNews-vectors-negative300.bin
        if (".bin" in vector_file):
            self.model = gensim.models.KeyedVectors.load_word2vec_format(vector_file, binary=True)
            self.data_set = data_set
            self.set_w2v_vectors_keys(number_of_words)
        elif(".txt" in vector_file):
        	set_glove_vectors_keys(vector_file)
        else:
            print("Error. You must use vector_file of type .bin (google) or .txt (glove)")
        if (data_set == 0):
            self.dataset_vectors = self.w2v_vectors
            self.dataset_keys = self.w2v_keys
        else:
            self.set_dataset_vectors_keys()


    def set_glove_vectors_keys(vector_file):
        #This function loads the set of glove vectors and keys, which will later be compared to the data_set
        # params --
        # vector_file: This file is either GoogleNews-vectors-negative300.bin or glove .txt file
        t_io = timeit.default_timer()
        glove_vectors = [] #This the vectorized word table.
        glove_keys = [] #This is the 100,000 word table but without the vector representations, just words. 'the', 'and', etc
        dataset_file = open(vector_file, 'r')
        file = dataset_file.readlines()
        for line in file:
            glove_vectors.append(line.split()[1:])
            glove_keys.append(line.split()[:1])

        glove_vectors /= np.linalg.norm(np.asarray(glove_vectors, dtype = np.float32), axis=1).reshape(-1, 1) #Numbers read in as string. Must be converted to floats.
        self.w2v_vectors = glove_vectors
        self.w2v_keys = glove_keys

    def set_w2v_vectors_keys(self, number_of_words):
        #This function loads the set of google vectors and keys, from a list of 1M words, which will later be compared to the data_set
        #params  ---
        # number_of_words : Specifies the size of the subset of the GoogleNews-vectors-negative300.bin that you want to compare to your dataset
        vectors = []
        keys = []
        for i in range (0,number_of_words):
            vectors.append(self.model.get_vector(self.model.index2entity[i]))
            keys.append((self.model.index2entity[i]))
        vectors/= np.linalg.norm((np.asarray(vectors, dtype = np.float32)), axis=1).reshape(-1, 1)
        self.w2v_vectors = vectors
        self.w2v_keys = keys
    def get_w2v_vectors(self):
        return self.w2v_vectors
    def get_w2v_keys(self):
        return self.w2v_keys

    def set_dataset_vectors_keys(self):
        # Vectorizes words in data_set that are found in glove or google
        # Returns the number of words in user data_set that were not vectorized
        dataset_vectors = []
        dataset_keys = []
        num_not_vectorized = 0
        for i in range (0, len(self.data_set)) :
                try:
                    dataset_vectors.append(self.model.get_vector(self.data_set[i]))
                    dataset_keys.append(self.data_set[i])

                except KeyError:
                    num_not_vectorized += 1
                    continue
        dataset_vectors  /= np.linalg.norm((np.asarray(dataset_vectors, dtype = np.float32)), axis=1).reshape(-1, 1)
        self.dataset_vectors = dataset_vectors
        self.dataset_keys = dataset_keys
        num_vectorized = len(self.data_set) - num_not_vectorized
        print("{} Words In Data Set Successfully Vectorized of {} Words".format(num_vectorized, len(self.data_set)))
        return num_not_vectorized
    def get_dataset_vectors(self):
        return self.dataset_vectors
    def get_dataset_keys(self):
        return self.dataset_keys
    


#############################################################################
#------------------------------LSH SECTION----------------------------#
    def linearScan_answerGenerator(self, train_vectors, queries):
        #Function description:
        #Returns the actual nearest neighbors of a set of queries by performing
        #a linear scan of the data_setself.

        #params ---
        #train_vectors: The subset of vectors from our vector_file that are used
        #to determine the optimal number of probes (default 90% accuracy)
        #queries: the word vectors used to determine the number of probes
        #needed to gain a search accuracy of 90% (default)
        print('Solving queries using linear scan')
        t1 = timeit.default_timer()
        answers = []
        for query in queries:
            answers.append(np.dot(train_vectors, query).argmax())
        t2 = timeit.default_timer()
        print('Done')
        print('Linear scan time: {} per query'.format((t2 - t1) / float(
            len(queries))))
        return answers


    def evaluate_number_of_probes(self, number_of_probes, query_object, answers, queries):
        #Function description:
        # Returns how accurate LSH is with given number of probes
        # as ratio of correct answers LSH got compared to all correct answers
        # (found by running linearScan_answerGenerator over queries)
        #params ---
        #number_of_probes: the total number of hash buckets over the LSH index checked for a given query
        #query_object: Essentialy the LSH index, specific object to Falconn package
        #queries: the word vectors used to determine the number of probes needed to gain a search accuracy of 90%
        #answers:the answers as to what the neareset neighbor is for the queries, used to determine the accuracy of a nearest neighbor searched.



        query_object.set_num_probes(number_of_probes)
        score = 0
        for (i, query) in enumerate(queries):
            if answers[i] in query_object.get_candidates_with_duplicates(query):
                score += 1
        return float(score) / len(queries)


    def probeGenerator(self, query_accuracy, number_of_probes, query_object, answers, queries, number_of_tables):
        # Returns number of probes that provide 90% accuracy to search.
        # params ---
        #number_of_probes: the total number of hash buckets over the LSH index checked for a given query
        #query_object: Essentialy the LSH index, specific object to Falconn package
        #answers:the answers as to what the neareset neighbor is for the queries, used to determine the accuracy of a nearest neighbor searched.
        #queries: the word vectors used to determine the number of probes needed to gain a search accuracy of 90%
        #number_of_tables:the number of hash_tables used for a given nearest neighbor search

        while True:
            accuracy = self.evaluate_number_of_probes(number_of_probes, query_object, answers, queries)
            print('{} -> {}'.format(number_of_probes, accuracy))
            if accuracy >= query_accuracy:
                 break
            number_of_probes = number_of_probes * 2
        if number_of_probes > number_of_tables:
            left = number_of_probes // 2
            right = number_of_probes
            while right - left > 1:
                number_of_probes = (left + right) // 2
                accuracy = self.evaluate_number_of_probes(number_of_probes, query_object, answers, queries)
                print('{} -> {}'.format(number_of_probes, accuracy))
                if accuracy >= query_accuracy:
                    right = number_of_probes
                else:
                    left = number_of_probes
            number_of_probes = right

        print('Done')
        print('{} probes'.format(number_of_probes))
        return number_of_probes

    def set_clustering_LSH_Index(self, number_of_queries, query_accuracy, number_of_tables, hash_bit):
        #Function defintion: Returns the LSH Index -- Read LSH for more information or
        # README.2

        #parameters
        #number_of_queries:The number of queries used to determine the number_of_probes
        #query_accuracy: Specifies the level of accuracy of the Index
        #Setting query_accuracy = 1 degenerates LSH index into linear search.
        #number_of_tables:the number of hash_tables used for a given nearest
        #neighbor search
        #hash_bit: Used to determine the number of hash functions. READ_ME for detail.
        print("Setting Clustering Index")

        queries = self.w2v_vectors[(len(self.w2v_vectors)-number_of_queries):]
        w2v_vectors = self.w2v_vectors[:(len(self.w2v_vectors)-number_of_queries)]

        #Normalize vectors
        center = np.mean(w2v_vectors, axis=0)
        w2v_vectors -= center
        queries -= center


        #perform linear scan to return correct answers
        answers = self.linearScan_answerGenerator(w2v_vectors, queries)

        #Set number of probes----
        print('Choosing number of probes')
        init_number_of_probes = 600
        # END -------

        #Parameters -----
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = len(w2v_vectors[0]) # = 50 for Glove6B.50d
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = number_of_tables
        params_cp.num_rotations = 1
        params_cp.seed = 5721840
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        falconn.compute_number_of_hash_functions(hash_bit, params_cp)
        # END ------

        #Constructing LSH Index -----
        print('Constructing the LSH Index')
        t1 = timeit.default_timer()
        table = falconn.LSHIndex(params_cp)
        table.setup(w2v_vectors)
        t2 = timeit.default_timer()
        print('Done')
        print('Construction time: {}'.format(t2 - t1))
        query_object = table.construct_query_object()
        number_of_probes = self.probeGenerator(query_accuracy, init_number_of_probes, query_object, answers, queries, number_of_tables)
        query_object.set_num_probes(number_of_probes)
        #--------

        # Performance Statistics
        t1 = timeit.default_timer()
        score = 0
        for (i, query) in enumerate(queries):
            if query_object.find_nearest_neighbor(query) == answers[i]:
                score += 1
        t2 = timeit.default_timer()
        print('Query time: {}'.format((t2 - t1) / len(queries)))
        print('Precision: {}'.format(float(score) / len(queries)))
        self.query_object = query_object
        print("Vectors Successfully Hashed. Clustering LSH Index Created")

    def get_clustering_LSH_index(self):
        #Returns LSH Index for clustering as opposed to the one used for hierarchical
        return self.query_object


#############################################################################
#------------------------------CLUSTERING SECTION----------------------------#
    def cluster_chained (self, m = 200, per_query_size = 5, hierarchical = 1,
        number_of_queries = 1000, query_accuracy = .9, number_of_tables = 160, hash_bit = 18):
        #Function: Please see below defintion for function description (line 286)
        #Params ---
        # m: The number of words per cluster
        # per_query_size: The length of each chain, how many words are searched to form the chain
        # hierarchical: The number of agglomerative clustering steps to do after cluster_chaining
        # number_of_queries: The number of vectors queried to determine the number of probes (see LSH section line 114)
        # query_accuracy: The desired accuracy of the LSH search setting = 1 degenerates into linear scan
        # number_of_tables: the number of tables used in each nearest neighbor search (see LSH section line 114)
        #hash_bit: Used to determine the strength of the hash_function see README.2 or LSH for more detail

        if (hierarchical == 0):
                return self.__cluster_chaining(m,per_query_size,
                        number_of_queries, query_accuracy, number_of_tables, hash_bit)
        elif isinstance(hierarchical, int):
            cluster = self.__cluster_chaining(m,per_query_size,
                    number_of_queries, query_accuracy, number_of_tables, hash_bit)
            cluster = self.heirarchical_cluster(cluster, query_accuracy, number_of_queries, number_of_tables, hash_bit)
            numHierarchical_clusterings = 0
            while (numHierarchical_clusterings < hierarchical-1):
                cluster = self.heirarchical_cluster(cluster, query_accuracy, number_of_queries, number_of_tables, hash_bit)
                numHierarchical_clusterings += 1
            return cluster
        else:
            print("Error. hierarchical parameter not correctly set. See READ_ME file for how cluster_chained() works.")
            return 1

    def __cluster_chaining(self, m, per_query_size,
        number_of_queries, query_accuracy, number_of_tables, hash_bit):
        # Personal Method modelled after human thinking. The cluster is built by first appending the cluster_size (hyper-parameter)/3 nearest neighbors (NN) of our orginal query.
        # The method then takes the 1st NN of the query and averages the 1st NN with the original query
        # This becomes our new query, 2 NN are generated from this new query (which is the avg of the original query and the 1st NN of the original query)
        # We now have a total cluster size of cluster_size/3 + 2 words. To construct our 3rd query, we take the WEIGHTED average of the original query and the orginal query's 2nd NN
        # We then run a NN search on this new query, appending 2 additional words to our cluster
        # We repeat the process, whereby for each new query we take a weighted average of our original query and its nth NN. As n grows (ie 5th NN, 10th NN of original query),
        # we progressively add more weight the original query when taking the average of our original query and its nth NN to generate the new query (query_n).
        # Now we have our new query, and run a NN search on it, appending 2 more NN.
        # We do this until we have reached the desired cluster size (default 200)

        # The idea behind the algorithm came from a thought experiment.
        # Problem: Say you are given a word (original_query) and tasked to find ~100 (cluster_size) words that are related to that original word(original_query)
        # Step 1: You come up with around 30 (cluster_size/3) words that are related to the orginal query (either synonyms, antonyms, etc). Let's call this list the original_list
        # Step 2: Then you are stuck. So you look at the 30 words that you came up with and begin thinking of words related to those 30 words in order to reach 100 words related to the original query.
        # Step 3: But you know that you cannot simply use the 30 words in the original_list to generate more words.  This is because you are ultimately trying to come up with 100 words related to the original_query
        # Step 4: So instead of just thinking of words related to the 30 words in the original_list. You consider the 30 words IN THE CONTEXT OF THE ORGINAL QUERY (weighted average of the original_query, and nth NN)
        # Step 5: You then generate words that are 1) related to the words within the orginal_list and 2) are in the context of the original_query
        # Step 6: You do this until you reach the desired cluster_size (100 words for example)

        # params --
        # m: The number of words per cluster
        # per_query_size: The length of each chain, how many words are searched to form the chain
        # hierarchical: The number of agglomerative clustering steps to do after cluster_chaining
        # number_of_queries: The number of vectors queried to determine the number of probes (see LSH section line 114)
        # query_accuracy: The desired accuracy of the LSH search setting = 1 degenerates into linear scan
        # number_of_tables: the number of tables used in each nearest neighbor search (see LSH section line 114)
        #hash_bit: Used to determine the strength of the hash_function see README.2 or LSH for more detail

        self.set_clustering_LSH_Index(number_of_queries, query_accuracy, number_of_tables, hash_bit)

        cluster = []
        financeNotRemoved = self.dataset_keys.copy()
        for i in range (0,len(self.dataset_vectors)):
            #print ("{} of {}".format(i, len(self.dataset_vectors)))
            if (self.dataset_keys[i] in financeNotRemoved):
                subcluster, subclusterNN = self.NNS_compiler(self.dataset_vectors[i], self.w2v_vectors, self.w2v_keys, m, per_query_size)
                for j in subclusterNN:
                    try:
                        financeNotRemoved.remove(self.w2v_keys[j]) #= list(filter(lambda finance: finance !=  google_keys[j], financeNotRemoved))
                    except ValueError:
                        continue
                cluster.append(subcluster)
            else:
                continue
        return cluster



    def NNS_builder(self, vector, m = 5):
        # Returns the 5 NN of a given query. query_1 = original query. query_2 = avg (query_1, 1st NN of query_1). query_3 = avg (2/1.5 * query_1, 2nd NN of query_1)
        # ...query_N = (N/1.5 * query_1, nth NN of query_1)
        NN_index = self.get_clustering_LSH_index().find_k_nearest_neighbors(vector, m)
        return NN_index
    def averager(self, w2v_vector1, w2v_vector2):
        #Simply takes average of w2v_vector1 and w2v_vector2
        return np.mean((w2v_vector1, w2v_vector2), axis = 0)
    def NNS_compiler(self, dataset_vector, w2v_vectors, w2v_keys, cluster_size, per_query_size):
        # Master Function For Cluster Chaining
        subcluster = [] #The cluster of words we are building. The main object returned by the function
        NNS_index = [] #List of NNs index
        ONNS_subindex = self.NNS_builder(dataset_vector, math.floor(cluster_size/(3))) #The list of indexes of the vectors returned by a nearest neighbor search on query_1 (original query)
        for j in ONNS_subindex:
             subcluster.append(self.w2v_keys[j]) #Appending the first cluster_size/3 NN generated by query_1 (our original query)
        for i in range (2, per_query_size + 1): #The main routine of the cluster chaining algorithm. It runs a NN-search on each query = avg (n/1.5*query _1, nth NN of query_1)
            #It runs a NN-search on (per_query_size+1-2) number of queries.
            NNS_subindex = self.NNS_builder(self.averager(w2v_vectors[ONNS_subindex[i]],(math.floor(i/1.5)*dataset_vector)), 2)#math.floor(cluster_size/(3*i)))
            for j in NNS_subindex:
                subcluster.append(w2v_keys[j])
                NNS_index.append(j)
        return list(set(subcluster)), NNS_index



    def cluster_vanilla(self, m, option = "vanilla",
    number_of_queries = 1000, query_accuracy = .9, number_of_tables = 160, hash_bit = 18):
        #Function description
        # option = 'vanilla' --- Basic agglomerative clustering ---
        # User specifies size of cluster, m. Function, beginning with first Word
        # finds the m nearest neighbors of the given word, then removes all of
        # the clustered words from the data_set. The function then moves to the
        # next words to be clustered and repeats. Until ~ number_of_words/m
        #clusters are generated.
        #option = 'disjoint' -- Disjoint agglomerative clustering --
        #Only clusters nearest neighbors within the data_set, whereas 'auto'
        #Can return nearest neighbors from the google set and not in the dataset
        #Just as in auto, once words have been clustered, they are no longer
        #considered in future nearest neighbor searches

        #Params ---
        # m: The number of words per cluster
        # option: see above function description (line 373)
        # number_of_queries: The number of vectors queried to determine the number of probes (see LSH section line 114)
        # query_accuracy: The desired accuracy of the LSH search setting = 1 degenerates into linear scan
        # number_of_tables: the number of tables used in each nearest neighbor search (see LSH section line 114)
        #hash_bit: Used to determine the strength of the hash_function see README.2 or LSH for more detail

        self.set_clustering_LSH_Index(number_of_queries, query_accuracy, number_of_tables, hash_bit)
        query_object = self.get_clustering_LSH_index()

        cluster = []
        datasetNotRemoved = self.dataset_keys.copy()

        remove_counter = 0 #Debugger to make sure words have been removed
        if (option == "vanilla"):
            for i in range(0,len(self.dataset_vectors)):
                #print ("{} of {}".format(i, len(self.dataset_vectors)))
                if (self.dataset_keys[i] in datasetNotRemoved):
                    subcluster = []
                    subclusterNN = self.get_clustering_LSH_index().find_k_nearest_neighbors(self.dataset_vectors[i], m)
                    for j in subclusterNN:
                        subcluster.append(self.w2v_keys[j])
                        try:
                            datasetNotRemoved.remove(self.w2v_keys[j]) #= list(filter(lambda finance: finance !=  google_keys[j], financeNotRemoved))
                        except ValueError:
                            continue
                    cluster.append(subcluster)
                else:
                    remove_counter +=1
                    continue
            if (remove_counter == 0):
                 print("Error. Dataset keys not successfully removed. See READ_ME file for how cluster_generator works.")
            return cluster
        elif (option == 'disjoint'):
            for i in range (0,len(self.dataset_vectors)):
                #print ("{} of {}".format(i, len(self.dataset_vectors)))
                if (self.dataset_keys[i] in datasetNotRemoved):
                    subcluster = []
                    subclusterNN = self.get_clustering_LSH_index().find_k_nearest_neighbors(self.dataset_vectors[i], m)
                    for j in subclusterNN:
                            if (self.w2v_keys[j] in datasetNotRemoved):
                                subcluster.append(self.w2v_keys[j])
                                datasetNotRemoved.remove(self.w2v_keys[j])
                            else:
                                continue
                    cluster.append(subcluster)
                else:
                    remove_counter +=1
                    continue
            if (remove_counter == 0):
                ("Error. Dataset keys not successfully removed. See READ_ME file for how cluster_generator works.")
            return cluster
        else:
            print("Error. Option parameter not correctly set. See READ_ME file for how cluster_generator works.")
            return 1


#############################################################################
#--------------------AGGLOMERATIVE CLUSTERING SECTION-----------------------#

    def __average_array_Generator(self, cluster):
        # takes the average of each vector
        average_array = []
#        if(distance == 'median'):
        for element in cluster:
            average_array.append(np.median(element, axis = 0))
        #if(distance == 'mean'):
    #        for element in cluster:
#                average_array.append(np.mean(element, axis = 0))
        #else:
    #        print("ERROR. Please use either 'mean or 'median' for distance parameter")
        #print('Cosine Similarity Metric Being Used.')
        #average_array /= np.linalg.norm(average_array, axis=0).reshape(-1, 1)
        print('Done')
        return average_array
    def __set_hierarchical_LSH_Index(self, cluster, number_of_tables, hash_bit):
        #Function defintion: Returns the LSH Index for hierarchical clustering
        # -- Read LSH for more information or README.2
        #params ---
        #cluster: the set of vectors wished to be clustered.
        # number_of_tables: the number of tables used in each nearest neighbor search (see LSH section line 114)
        #hash_bit: Used to determine the strength of the hash_function see README.2 or LSH for more detail
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = len(cluster[0])
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = number_of_tables
        params_cp.num_rotations = 1 #Parameter associated with crosspolytope see Falconnn for more
        params_cp.seed = 5721840
    # we want to use all the available threads to set up
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

        hash_bit = math.floor(math.log(len(cluster),2))
        # we build 32-bit hashes so that each table has
        # 2^32 bins; this is a good choise since 2^32 is of the same
        # order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(hash_bit, params_cp) #Look at typical number of hash functions
        #Figure out how number of hashfunctions are determined.

        print('Constructing the LSH Index For Cluster Combine Method.')
        t1 = timeit.default_timer()
        table = falconn.LSHIndex(params_cp)
        table.setup(cluster)
        t2 = timeit.default_timer()
        print('Done')
        print('Construction time: {}'.format(t2 - t1))

        self.hierarchical_LSH_Index = table.construct_query_object()
    def get_hierarchical_LSH_Index(self):
        return self.hierarchical_LSH_Index

    def heirarchical_cluster(self, cluster, query_accuracy, number_of_queries, number_of_tables, hash_bit):
        # Function brief descrition: takes singletons, clusters NN into tuples, averages
        # tuples, clusters tuples of averages and repeats until desired cluster size
        # graphically 1 (* * * * * * *) ->  2 ((*,*), (*,*), (*,*), (*) )
        # -> 3 ((a), (a), (a), (a=*)) -> repeat
        # params ---
        #cluster: the set of singletons wishing to be clustered
        # number_of_queries: The number of vectors queried to determine the number of probes (see LSH section line 114)
        # query_accuracy: The desired accuracy of the LSH search setting = 1 degenerates into linear scan
        # number_of_tables: the number of tables used in each nearest neighbor search (see LSH section line 114)
        #hash_bit: Used to determine the strength of the hash_function see README.2 or LSH for more detail

        vclusters = []
        for clust in cluster:
            vcluster = []
            for key in clust:
                vcluster.append(self.model.get_vector(key))
            vclusters.append(vcluster)
        cluster_vectors = vclusters.copy()
        cluster_keys = cluster.copy()

        average_array = self.__average_array_Generator(cluster_vectors)

        number_of_queries = math.floor(len(average_array)/10)
        queries = average_array[len(average_array) - number_of_queries:]
        average_array = average_array[:len(average_array) - number_of_queries]
        average_array = np.asarray(average_array,  dtype = np.float32)
        #Centering Array
        center = np.mean(average_array, axis=0)
        average_array -= center
        queries -= center

        #Constructing LSH Index
        number_of_tables = 160
        self.__set_hierarchical_LSH_Index(average_array, number_of_tables, hash_bit)
        query_object = self.get_hierarchical_LSH_Index()

        #Determing optimal number of probes for given query_accuracy (.9 recommended)
        answers = self.linearScan_answerGenerator (average_array, queries)
        number_of_probes = number_of_tables
        number_of_probes = self.probeGenerator(query_accuracy, number_of_probes, query_object, answers, queries, number_of_tables)
        query_object.set_num_probes(number_of_probes)

        print("Generating Clusters")
        t1 = timeit.default_timer()

        searched = []
        cluster_list = []
        cluster_key_list = []
        for index in range(0,len(average_array)):
            new_cluster = []
            new_cluster_key = []
            if(index not in searched):
                neighbor_indices = query_object.find_k_nearest_neighbors(average_array[index], 2)
                for subindex in neighbor_indices:
                    if(subindex not in searched):
                        searched.append(subindex)
                        for subsubindex in range(0, len(cluster_keys[subindex])):
                           new_cluster_key.append(cluster_keys[subindex][subsubindex])
                cluster_key_list.append(new_cluster_key)
        t2 = timeit.default_timer()
        print('Done')
        print('Cluster time: {}'.format(t2 - t1))
        del average_array
        print("Hierarchical Cluster Sucessfully Generated")
        return cluster_key_list

##############################################################################
# ------------ Auxiliary Functions. Will add more in v2 ---------------------#
    def write_to_text(self, cluster, filepath):
            # Prints the set of clusters to textfile
            # param ---
            # cluster: the set of clusters (array(cluster1, cluster2,...,(clusterN)))
            # filepath:the user-specified location that the .txt file will be
            textfile = open(filepath, 'w')
            for cluster in cluster:
                textfile.write(np.array_str(np.ravel(cluster)))
                textfile.write('\n')
            textfile.close()
            return filepath
