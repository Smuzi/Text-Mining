import sys
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix,lil_matrix


def get_new_centroids(X,closest_cluster):
	num_of_clusters = np.amax(closest_cluster)+1
	#lil_matrix fast for calculating mean
	temp_matrix = lil_matrix((num_of_clusters,X.shape[1]))			#num_of..x X.cols sparse matrix
	for i in range(num_of_clusters):
		relevant_cluster = X[closest_cluster == i]
		if relevant_cluster.shape[0]!=0:
			temp_matrix[i] = relevant_cluster.mean(axis=0)			#calculate new mean of centroids
	centroids = csr_matrix(temp_matrix)								#sparse row matrix
	return centroids
	
def random_numbers(S,N):
	#generate S different numbers from 0 to N
	random.seed()
	Range = np.arange(S)
	for i in range(0,S):
		num = random.randint(0,N)
		while num in Range:
			num = random.randint(0,N)
		Range[i] = num
	return Range

def kmean(X,k,initial_centroids=None):
	
	if initial_centroids==None:
		#initial random seeds and centroids:
		seeds_number = random_numbers(k,X.shape[0]-1) 					#shape[0] returns number of rows
		#print "rows:",X.shape[0]
		initial_centroids = X[seeds_number]
	
	#create clusters:
	distance_matrix = pairwise_distances(X,initial_centroids)		#calculate distance from each centroid
	closest_cluster_id = distance_matrix.argmin(axis=1) 			#get smallest distance
	return kmean_body(X,closest_cluster_id)

def kmean_body(X,closest_cluster):
	counter = 0
	closest_cluster_prev = closest_cluster
	while True:
		centroids = get_new_centroids(X,closest_cluster_prev)
		distances_matrix = pairwise_distances(X,centroids)
		closest_cluster_new = distances_matrix.argmin(axis=1)
		if counter>=20 or np.count_nonzero(closest_cluster_new!=closest_cluster_prev)<=0.01*X.shape[0]:
			return closest_cluster_prev
		closest_cluster_prev = closest_cluster_new
		counter +=1
def bipartite_clustering(D2W,word_cluster_num,doc_cluster_num):
	W2D = D2W.transpose()
	W2WC = kmean(W2D,word_cluster_num)
	
	for loop in range(20):
		#D2WC = D2W.dot(transform_from_index_array(W2WC,W2WC.size,word_cluster_num))
		#print D2WC
		new_centroids = get_new_centroids(W2D,W2WC)
		new_distance_matrix = pairwise_distances(W2D,new_centroids) #how to calculate distance? maybe 1-matrix?
		
		D2WC = D2W.dot(new_distance_matrix)
		if loop==0:
			D2DC = kmean(D2WC,doc_cluster_num)
		else:
			new_centroids = get_new_centroids(D2WC,D2DC)
			D2DC = kmean(D2WC,doc_cluster_num,new_centroids)
		
		new_centroids = get_new_centroids(D2W,D2DC)
		new_distance_matrix = pairwise_distances(D2W,new_centroids) #how to calculate distance? maybe 1-matrix?
		
		W2DC = W2D.dot(new_distance_matrix)
		new_centroids = get_new_centroids(W2DC,W2WC)
		W2WC = kmean(W2DC,word_cluster_num,new_centroids)
	return D2DC,W2WC

def main(wc,dc,filename,docOutput,wordOutput):
	num_WC = wc
	num_DC = dc
	
	with open(filename) as f:
		documents = f.readlines()	
	rows = []
	cols = []
	data = []
	for document_id,document in enumerate(documents):
		for pair in document.strip().split(" "):
			pair_array = pair.split(":")
			word_id = pair_array[0]
			word_tf = pair_array[1]
			rows.append(document_id)
			cols.append(word_id)
			data.append(word_tf)
	
	number_of_words = np.amax(np.array(cols).astype(int)) + 1
	number_of_documents = document_id + 1
	
	Doc2Word = csr_matrix((np.array(data).astype(float),(np.array(rows),np.array(cols))),shape=(number_of_documents,number_of_words),dtype=float)
	Word2Doc = Doc2Word.transpose(copy=True)
	Doc2WordC,Word2DocC = bipartite_clustering(Doc2Word,num_DC,num_WC)

	DC_out = ['%i %i\n' % (doc_id, cluster_id) for doc_id, cluster_id in enumerate(Doc2WordC)]
	WC_out = ['%i %i\n' % (doc_id, cluster_id) for doc_id, cluster_id in enumerate(Word2DocC)]
	
	with open(docOutput,'w') as f:
		f.writelines(DC_out)
	with open(wordOutput,'w') as f:
		f.writelines(WC_out)
	
	return '0'
