########### The core of the Textual Factors Model ###########
# it inlcude three parts of the model:
# 1. the NeighborFinder class that is used to find the nearest neighbors of the embeddings
# 2. EmbeddingCluster class that is used to cluster the embeddings using the near neighbors
# 3. TextualFactors class that is used to construct the textual factors within each cluster
# the textual factors include the LSA topics and the LDA topics
# author: zhu,wu
# date: 2024-04-04

import faiss
import numpy as np
import pandas as pd 
class NeighborFinder:
    """"this class is used to find the nearest neighbors of the embeddings using 
    the brute-force search and the LSH index
    """
    def __init__(self, embeddings,
                 random_state=42,num_queries = 1000):
        # first normalize the embeddings 
        self.embeddings = embeddings/np.linalg.norm(embeddings,axis = 1).reshape(-1,1)
        self.dimension = embeddings.shape[1]
        # Initialize the brute-force index and add embeddings
        self.brutal_index = faiss.IndexFlatL2(self.dimension)
        self.brutal_index.add(embeddings)
        # set the random state for reproducibility
        self.random_state = random_state
        self.num_queries = num_queries # the number of queries used to evaluate the index accuracy 
        
        # Initialize the LSH index
        self.lsh_index = self.create_lsh_index()
    
    def create_lsh_index(self, n_bits_per_hash_table=8, n_hash_tables=4):
        """
        Initialize and return the LSH index after adding the embeddings.
        """
        self.n_bits_per_hash_table = n_bits_per_hash_table
        self.n_hash_tables = n_hash_tables
        lsh_index = faiss.IndexLSH(self.dimension, self.n_bits_per_hash_table * self.n_hash_tables)
        lsh_index.add(self.embeddings)  # Add embeddings to the LSH index
        return lsh_index
    
    def find_neighbors(self, index, queries, k=2):
        """
        Find the top k nearest neighbors using the specified index.
        Returns only the nearest neighbors, excluding the query itself if included.
        """
        D, I = index.search(queries, k)
        # Exclude the query itself if it's included in the nearest neighbors
        if k > 1:
            return D[:, 1], I[:, 1]
        else:
            return D, I
        
    def eval_index_accuracy(self,index,k = 2):
        """evaluate the accuracy of the index finding the ground truth neighbors"""
        # 1. select random queries
        np.random.seed(self.random_state)
        num_queries = self.num_queries  
        query_indexes = np.random.choice(len(self.embeddings),num_queries,replace = False,
                                         )
        # 2. get the ground truth neighbors for the queries 
        _, ground_truth = self.find_neighbors(self.brutal_index,self.embeddings[query_indexes],k = k)
        # 3. get the estimated neighbors determined by the index 
        _, estimated = self.find_neighbors(index,self.embeddings[query_indexes],k = k)
        # 4. compute the accuracy of the index
        accuracy = np.mean(ground_truth == estimated)
        return accuracy
    
    def optimize_lsh_hyperparameters(self, target_accuracy, max_trials=20, k=2):
        for trial in range(6, max_trials + 1):
            n_bits_per_hash_table = 2**trial   # Example strategy: Increase bits per table
            for n_hash_tables in range(1, trial):
                n_hash_tables = 2**n_hash_tables  # Example strategy: Increase number of tables
                temp_index = self.create_lsh_index(n_bits_per_hash_table, n_hash_tables)
                accuracy = self.eval_index_accuracy(temp_index, k=k)
                print(f"Trial {trial}: Accuracy = {accuracy:.2f}, Bits per Table = {n_bits_per_hash_table}, Hash Tables = {n_hash_tables}")
                if accuracy >= target_accuracy:
                    print("Desired accuracy achieved. Optimizing stopped.")
                    self.lsh_index = temp_index
                    print(f"Final LSH index: Bits per Table = {n_bits_per_hash_table}, Hash Tables = {n_hash_tables}")
                    return True, {"Accuracy": accuracy, "Bits per Table": n_bits_per_hash_table, "Hash Tables": n_hash_tables}
                
    
                
class EmbeddingCluster():
    def __init__(self,neighborfinde_model,neighbor_alg = 'lsh'):
        """neighborfinde_model: the model that is used to find the neighbors
        neighborfinde_model: NeighborFinder object which is used to find the near neighbors. the default is the lsh index
        you can also use the brute force index to find the near neighbors
        neighbor_alg: the algorithm used to find the near neighbors, the default is the lsh algorithm
        """
        self.model = neighborfinde_model
        self.neighbor_alg = neighbor_alg # set the algorithm used to find the near neighbors
        self.embeddings = self.model.embeddings.astype(np.float32)
        if neighbor_alg == 'lsh':
            self.index = self.model.lsh_index
        elif neighbor_alg == 'brutal':
            self.index = self.model.brutal_index
        self.dimension = self.model.dimension
    
    def find_topk(self,query_point,k = 10,similarity_threshold = 0.3):
        """find the approximate top k near neighbors of a query point"""
        D,I = self.index.search(query_point.reshape(1,-1),k)
        I = I[0]
        if self.neighbor_alg == 'lsh':
            for idx,i in enumerate(I):
                similarity = np.dot(self.embeddings[i],query_point)
                # print(i,similarity)
                if similarity < similarity_threshold:
                    return I[:idx]
        return I
        # return I # return the index of the near neighbors including the query point
    
    def sequentialcluster(self,cluster_size = 50):
        """perform sequential clustering of words based on the index:lsh or brutal 
        Returns:
            List of clusters, with each cluster being a list of indices corresponding to word embeddings.
        """
        Notvisited = list(range(len(self.embeddings))) # initialize with indices of all points
        clusters = []
        while Notvisited:
            queryIndex = Notvisited[0]  # Take the head of pointsNotVisited
            queryPoint = self.embeddings[queryIndex]
            # Find the approximate near neighbors including the query point itself
            ResultIndices = self.find_topk(queryPoint,k = cluster_size)
            # Form a new cluster with points that are both in ResultIndices and pointsNotVisited
            newCluster = [index for index in ResultIndices if index in Notvisited]
            clusters.append(newCluster)
            # Remove the indices of the new cluster from pointsNotVisited
            Notvisited = [index for index in Notvisited if index not in newCluster]
        return clusters
    
    def heuristic_cluster(self,initial_seeds,cluster_size = 50):
        """performe herustical clustering of words based on initial seeds for clustering
        initial_seeds: the indices of the initial seeds for clustering
        cluster_size: the size of the cluster
        """
        Notvisited = list(range(len(self.embeddings))) # initialize with indices of all points
        clusters = []
        
        # initialize clusters with seeds 
        for seed in initial_seeds:
            if seed in Notvisited:
                queryPoint = self.embeddings[seed]
                resultsIndices = self.find_topk(queryPoint,k = cluster_size)
                newCluster = [index for index in resultsIndices if index in Notvisited]
                clusters.append(newCluster)
                Notvisited = [index for index in Notvisited if index not in newCluster]
        
        # Continue with the rest of the points 
        while Notvisited:
            queryIndex = Notvisited[0]
            queryPoint = self.embeddings[queryIndex]
            ResultIndices = self.find_topk(queryPoint, k=cluster_size)
            newCluster = [index for index in ResultIndices if index in Notvisited]
            clusters.append(newCluster)
            Notvisited = [index for index in Notvisited if index not in newCluster]
        return clusters

    def set_index(self, embeddings):
        """Setup the index."""
        # Assuming NeighborFinder is a custom class that you've defined to manage the FAISS index
        neighborfinder = NeighborFinder(embeddings, random_state=42, num_queries=1000)
        if self.neighbor_alg == 'lsh':
            neighborfinder.optimize_lsh_hyperparameters(0.9)
            centroid_index = neighborfinder.lsh_index
        else:
            centroid_index = neighborfinder.brutal_index
        return centroid_index

    def hierarchical_cluster(self, K,similarity_threshold = 0.5):
        """Perform hierarchical clustering until K clusters are left, merging only sufficiently similar clusters."""
        clusters = [[i] for i in range(len(self.embeddings))]  # Initialize each point as its own cluster
        
        while len(clusters) > K:
            initial_num_clusters = len(clusters)
            centroids = np.array([np.mean(self.embeddings[cluster], axis=0) for cluster in clusters])
            # normalize the centroids
            centroids = centroids / np.linalg.norm(centroids, axis=1).reshape(-1, 1)
            # if the number of clusters is large, use the lsh index to find the near neighbors
            # else just use the brutal force to find the near neighbors
            if initial_num_clusters > 20000:
                centroid_index = self.set_index(centroids)
            else:
                self.neighbor_alg = 'brutal'
                centroid_index = self.set_index(centroids)
            D, I = centroid_index.search(centroids, 2)  # Query for 2 neighbors: self and nearest
            
            merged_indices = set() 
            new_clusters = [] 
            for i in range(len(clusters)):
                if i in merged_indices:
                    continue  # Skip clusters that have already been merged
                
                nearest = I[i, 1]
                # calculate the cosine similarity between the centroids
                similarity = np.dot(centroids[i], centroids[nearest])
                if nearest not in merged_indices and i != nearest and similarity > similarity_threshold :  # Check if distance is within threshold
                    merged_cluster = clusters[i] + clusters[nearest]
                    new_clusters.append(merged_cluster)
                    merged_indices.add(i)
                    merged_indices.add(nearest)
                elif i != nearest:  # Keep cluster i if nearest is already merged or distance is above threshold
                    new_clusters.append(clusters[i])
                    merged_indices.add(i)

           # Handle any clusters not merged in this iteration
            for idx, cluster in enumerate(clusters):
                if idx not in merged_indices:
                    new_clusters.append(cluster)
            clusters = new_clusters
            num_merged_clusters = len(clusters)
            # if no further merged clusters, break the loop
            print(f"the number of merged clusters is : {initial_num_clusters - num_merged_clusters}")
            if initial_num_clusters - num_merged_clusters < 10:
                break
        return clusters
    def cluster_word_map(self,clusters):
        """generate a mapping of the words to the clusters"""
        # a cluster-word-map, key: cluster_id, value: word id in the cluster
        cluster_words = {i: clusters[i] for i in range(len(clusters))}
        # a word-cluster-map, key: word id, value: cluster id
        word_cluster = {word_id: cluster_id for cluster_id, word_ids in cluster_words.items() for word_id in word_ids}
        return cluster_words,word_cluster  
    
from sklearn.decomposition import TruncatedSVD
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from tqdm.auto import tqdm

class TextualFactors():
    """this class is construct the textual factors , that is, we construct the topic analysis within each embedding 
    clusters. 
    We need two datasets:
    1. the word-cluster dataset which contains the cluster id for each word, it includes two columns: cluster_id and word
    2. the document-word dataset, each row contains the document id, the word, and the count of the word in the document
    """
    def __init__(self,document_word_data,word_cluster_data):
        """
        document_word_data: a dataframe, with document_id as the index, and the words as the columns, the count of words as vlaues 
        word_cluster_data:  a dataframe, with two columns: cluster_id and word
        """
        self.document_word_data = document_word_data
        self.word_cluster_data = word_cluster_data
    
    def L1topic_importance(self,cluster_type = 'sequential_clsuter'):
        """this method to calculate the L1 topic importance for each cluster as defined in algorithm 3:
        which is just the total number of words frequency in the cluster
        """
        df = self.word_cluster_data[['ngram',cluster_type]]
        df.rename(columns = {cluster_type:'cluster'},inplace = True)
        # 1. get the unique clusters id 
        cluster_ids = df['cluster'].unique().tolist()
        cluster_ids.sort()
        
        # 2. create the document-word matrix for each cluster 
        document_topics = {}
        topics_words = {}
        singular_values = {}
        topic_importances = {}
        for cluster_id in cluster_ids:
            # 2.1 get the word support for the cluster 
            cluster = df[df['cluster'] == cluster_id]['ngram'].tolist()
            # 2.2 get the doc_word_matrix 
            doc_word_matrix = self.document_word_matrix(cluster)
            # 2.3 calculate the L1 topic importance for the cluster which is just the word frequency in the cluster 
            document_topic = doc_word_matrix.T.sum().to_dict()
            document_topics[cluster_id] = document_topic
            # 2.4 calculate the topic which is just the word frequency in the cluster 
            word_freq = doc_word_matrix.sum().sort_values(ascending = False)
            word_freq = word_freq/word_freq.sum()
            topic_word_dict = word_freq.to_dict()
            topics_words[cluster_id] = topic_word_dict
            topic_importances[cluster_id] = word_freq.sum()
        return document_topics,topics_words,topic_importances
    
    # step 1: create a document-word matrix 
    def document_word_matrix(self,cluster):
        """create a document-word matrix from the dataframe with word support on the cluster
        cluster: the cluster list contains the list of words in the cluster
        returns:
            doc_word_matrix: the documet-word matrix with documents as index and words as columns
        """
        df = self.document_word_data.copy()
        df = df[df['ngram'].isin(cluster)]
        # Step 1: Pivot the table to get the desired document-word matrix
        # Use 'document' as index, 'ngram' as columns, and 'count' as values
        doc_word_matrix = df.pivot_table(index='document', columns='ngram', values='count', aggfunc='sum')
        # Step 2: Fill missing values with zeros
        doc_word_matrix = doc_word_matrix.fillna(0)
        return doc_word_matrix

    # step 2: perform SVD on the document-word matrix
    def perform_svd(self,doc_word_df, n_components=2):
        """
        Performs SVD on the document-word matrix and returns the document-topic distribution
        for the first topic, the topic-word distribution for the first topic as a sorted dictionary
        by weight in descending order, and the first two singular values.
        
        :param doc_word_df: DataFrame with documents as rows and words as columns
        :param n_components: Number of components to keep, set to 2 for extracting the first two singular values
        :return: A tuple of (document-topic distribution for the first topic,
                sorted topic-word distribution dictionary for the first topic,
                first singular value, second singular value)
        """
        # Convert DataFrame to numpy array
        matrix = doc_word_df.values
        
        # Perform SVD
        svd = TruncatedSVD(n_components=n_components)
        U = svd.fit_transform(matrix)  # Document-topic distribution
        Sigma = svd.singular_values_   # Singular values
        VT = svd.components_           # Topic-word distribution
        
        # Extract the document-topic distribution for the first topic
        first_topic_document_distribution = U[:, 0]
        second_topic_document_distribution = U[:, 1] if n_components > 1 else None
        # create the document-topic dictionary for the first topic
        first_document_topic_dict = {doc: weight for doc, weight in zip(doc_word_df.index, first_topic_document_distribution)}
        second_document_topic_dict = {doc: weight for doc, weight in zip(doc_word_df.index, second_topic_document_distribution)} if n_components > 1 else None
        
        # Extract the topic-word distribution for the first topic
        first_topic_word_distribution = VT[0, :]
        second_topic_word_distribution = VT[1, :] if n_components > 1 else None

        # Create a sorted dictionary for the topic-word distribution
        words = doc_word_df.columns
        first_topic_word_dict = {word: weight for word, weight in zip(words, first_topic_word_distribution)}
        first_topic_word_dict = dict(sorted(first_topic_word_dict.items(), key=lambda item: item[1], reverse=True))
        
        second_topic_word_dict = {word: weight for word, weight in zip(words, second_topic_word_distribution)} if n_components > 1 else None
        second_topic_word_dict = dict(sorted(second_topic_word_dict.items(), key=lambda item: item[1], reverse=True) ) if n_components > 1 else None
        
        # Extract the first two singular values
        leading_singular_value = Sigma[0]
        second_singular_value = Sigma[1] if len(Sigma) > 1 else None
        # calcualte the topic importance 
        first_topic_importance = np.abs(first_topic_document_distribution).sum()*np.abs(leading_singular_value)
        # first_topic_importance = np.abs(svd.explained_variance_ratio_[0])
        second_topic_importance = np.abs(second_topic_document_distribution).sum()*np.abs(second_singular_value) if n_components > 1 else None
    
        results = {'first_document_topic_dict':first_document_topic_dict,
                   'second_document_topic_dict':second_document_topic_dict,
                   'first_topic_word_dict':first_topic_word_dict,
                   'second_topic_word_dict':second_topic_word_dict,
                   'leading_singular_value':leading_singular_value,
                   'second_singular_value':second_singular_value,
                   'first_topic_importance':first_topic_importance,
                   'second_topic_importance':second_topic_importance}
        return results
    
    def lsa_topics(self,cluster_type = 'sequential_cluster',n_topics = 2):
        """this method is to get the lsa topics for each cluster in the embedding clusters
        n_topics: the number of topics to extract from the document-word matrix within each cluster
        """
        df = self.word_cluster_data[['ngram',cluster_type]]
        df.rename(columns = {cluster_type:'cluster'},inplace = True)
        # 1. get the unique clusters id 
        cluster_ids = df['cluster'].unique().tolist()
        cluster_ids.sort()
        # 2. create the document-word matrix for each cluster 
        first_document_topics = {}
        second_document_topics = {}
        first_topics_words = {}
        second_topics_words = {}
        singular_values = {}
        topic_importances = {}
        for cluster_id in tqdm(cluster_ids):
            # 2.1 get the word support for the cluster 
            cluster = df[df['cluster'] == cluster_id]['ngram'].tolist()
            if len(cluster) < 5:
                continue
            try:
                # 2.2 get the doc_word_matrix 
                doc_word_matrix = self.document_word_matrix(cluster)
                # 2.3 perform SVD on the doc_word_matrix to get the document-topic distribution, top-word distribution, and singular values
                results = self.perform_svd(doc_word_matrix,n_components = n_topics)
                first_document_topics[cluster_id] = results['first_document_topic_dict']
                second_document_topics[cluster_id] = results['second_document_topic_dict']
                first_topics_words[cluster_id] = results['first_topic_word_dict']
                second_topics_words[cluster_id] = results['second_topic_word_dict']
                singular_values[cluster_id] = [results['leading_singular_value'],results['second_singular_value']]
                topic_importances[cluster_id] = [results['first_topic_importance'],results['second_topic_importance']]
                # print(f"the cluster {cluster_id} has been processed")
            except:
                continue
        return first_document_topics,second_document_topics,first_topics_words,second_topics_words,singular_values,topic_importances
    
    def lda_topics(self,cluster_type = 'sequential_cluster'):
        """this method is to get the lda topics for each cluster in the embedding clusters"""
        df = self.word_cluster_data[['ngram',cluster_type]]
        df.rename(columns = {cluster_type:'cluster'},inplace = True)
        # 1. get the unique clusters id 
        cluster_ids = df['cluster'].unique().tolist()
        cluster_ids.sort()
        # 2. create the document-word matrix for each cluster 
        document_topics = {}
        topics_words = {}
        topic_importance_ratio = {} # record the ratio of the most important topic to the second most important topic
        topic_importances = {}
        for cluster_id in tqdm(cluster_ids):
            # 2.1 get the word support for the cluster 
            cluster = df[df['cluster'] == cluster_id]['ngram'].tolist()
            doc_word_matrix = self.document_word_data[self.document_word_data['ngram'].isin(cluster)]
            # Assuming `df` is your initial DataFrame
            prepared_documents = self.prepare_data_for_lda(doc_word_matrix)
            most_important_topic_words,topic_importance,doc_topic_dist_most_important,importance_ratio = self.perform_lda_analysis(prepared_documents)
            document_topics[cluster_id] = doc_topic_dist_most_important
            topics_words[cluster_id] = most_important_topic_words
            topic_importances[cluster_id] = topic_importance
            topic_importance_ratio[cluster_id] = importance_ratio
            # print(f"the cluster {cluster_id} has been processed")
        return document_topics,topics_words,topic_importances,topic_importance_ratio
            
    def prepare_data_for_lda(self,df):
        """
        Transforms the DataFrame from long to wide format, suitable for LDA analysis.
        
        :param df: DataFrame with columns 'document', 'ngram', and 'count'.
        :return: A list of documents, where each document is represented as a list of words (with repetition according to frequency).
        """
        # Pivot table to document-word matrix
        doc_word_matrix = df.pivot_table(index='document', columns='ngram', values='count', fill_value=0)
        
        # Convert DataFrame to list of words for each document, respecting word frequencies
        documents = []
        for _, row in doc_word_matrix.iterrows():
            doc = []
            for word, frequency in row.items():  # Use .items() for Series objects
                doc.extend([word] * int(frequency))  # Repeat word according to its frequency
            documents.append(doc)
        return documents


    def perform_lda_analysis(self,documents, num_topics=10, passes=15, coherence='c_v'):
        """
        Performs LDA analysis on the prepared data, focusing on the weight of the most important topic for each document,
        and reports the ratio of the most important topic to the second most important topic.
        """
        # Create a dictionary and corpus for LDA
        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        
        # Perform LDA
        lda_model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        
        # Document-topic distribution with minimum_probability=0 for consistency
        doc_topic_distributions = [lda_model.get_document_topics(bow, minimum_probability=0) for bow in corpus]
        
        # Calculate average topic weights across all documents
        topic_weights = np.zeros(num_topics)
        for doc in doc_topic_distributions:
            for topic, weight in doc:
                topic_weights[topic] += weight
        topic_weights /= len(doc_topic_distributions)
        
        # Identify the most and second most important topics
        sorted_indices = np.argsort(topic_weights)[::-1]  # Descending order
        most_important_topic = sorted_indices[0]
        second_most_important_topic = sorted_indices[1]
        
        # Calculate the ratio of topic weights of the most important topic to the second most important topic
        importance_ratio = topic_weights[most_important_topic] / topic_weights[second_most_important_topic]
        
        # Extract the weight for the most important topic for each document
        doc_topic_dist_most_important = []
        for dist in doc_topic_distributions:
            topic_weight_dict = dict(dist)
            weight = topic_weight_dict.get(most_important_topic, 0)  # Default to 0 if topic not present
            doc_topic_dist_most_important.append(weight)
        
        # Extract the topic-word distribution for the most important topic
        most_important_topic_words = lda_model.show_topic(most_important_topic, topn=10)
        
        # Calculate coherence score
        coherence_model = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence=coherence)
        coherence_score = coherence_model.get_coherence()
        
        # Return the results including the importance ratio
        return (most_important_topic_words, topic_weights[most_important_topic], doc_topic_dist_most_important, importance_ratio)
    

def transfer_document_topics(document_topics):
    """transfer the dictionary document_topics to a dataframe"""
    pass

    for idx,cluster_id in enumerate(document_topics.keys()):
        if idx == 0:
            document_ids,topic_loadings = list(document_topics[cluster_id].keys()),list(document_topics[cluster_id].values())
            results = pd.DataFrame({'document':document_ids,f'topic_loading_{cluster_id}':topic_loadings})
        else:
            document_ids,topic_loadings = list(document_topics[cluster_id].keys()),list(document_topics[cluster_id].values())
            temp = pd.DataFrame({'document':document_ids,f'topic_loading_{cluster_id}':topic_loadings})
            results = pd.merge(results,temp,on = 'document',how = 'outer')
        print(f"the cluster {cluster_id} has been processed")
    
    # fill the na values with 0
    results.fillna(0,inplace = True)
    return results    

def transfer_topic_words(topics_words):
    """transfer the dictionary topics_words to a dataframe"""
    topic, topic_distribution = list(topics_words.keys()),list(topics_words.values())
    return pd.DataFrame({'topic':topic,'topic_distribution':topic_distribution})


def transfer_sigular_values(singular_values):
    """transfer the singular values into a dataframe with key as the cluster, and value as the leading and 
    second singular values"""
    cluster_id = list(singular_values.keys())
    singular_values = list(singular_values.values())
    leading_singular = [x[0] for x in singular_values]
    second_singular = [x[1] for x in singular_values]
    return pd.DataFrame({'cluster':cluster_id,'leading_singular':leading_singular,'second_singular':second_singular})

def transfer_topic_importances(topic_importances):
    """transfer the topic_importances into a dataframe with key as the cluster, and value as the leading and second importatn
    topic importances within the cluster"""
    cluster_id = list(topic_importances.keys())
    topic_importances = list(topic_importances.values())
    leading_importance = [x[0] for x in topic_importances]
    second_importance = [x[1] for x in topic_importances]
    return pd.DataFrame({'cluster':cluster_id,'leading_importance':leading_importance,'second_importance':second_importance})
