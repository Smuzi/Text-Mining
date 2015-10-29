import argparse
import sys
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import normalize
import time
import math
from bpc import bipartite_clustering,get_new_centroids
import pickle

def create_data_matrix(filename):
	#training data format: "MovieID","UserID","Rating","RatingDate"
	with open(filename,'r') as f:
		split_rows = map(lambda line:line.split(','),f.readlines())
	rating_data = map(lambda x:int(x[2]),split_rows)
	user_movie_pairs = [map(lambda x:int(x[1]),split_rows),map(lambda y:int(y[0]),split_rows)]
	num_of_users = max(user_movie_pairs[0])+1
	num_of_movies = max(user_movie_pairs[1])+1
	return csr_matrix((rating_data,user_movie_pairs),shape=(num_of_users,num_of_movies),dtype=float)
	
def read_query(filename):
	#The development set consists of movie-user pairs, without a rating. format:
	#"MovieID","UserID"
	with open(filename,'r') as f:
		split_rows = map(lambda line: line.split(','),f.readlines())
	queries = map(lambda x: (int(x[1]),int(x[0])), split_rows)
	return queries
	
def knn(X,k,row_num,col_num,similarity,weighted=False,exclude_self=True):
	#get the similarity row as an array:
	similarity = similarity[row_num].todense().A1
	#exclude user_i as one of the knn:
	if exclude_self:
		similarity[row_num] = 0
		
	#Perform an indirect partition. int index to partition by. 
	#The kth element will be in its final sorted position and all 
	#smaller elements will be moved before it and all larger elements behind it. 
	k_nearest_rows = np.argpartition(similarity,similarity.size-k)[-k:]
	
	#score calculation:
	item = X[k_nearest_rows,col_num].todense().A1 #
	sum_self_scores = np.sum(similarity[row_num])
	if weighted and sum_self_scores!=0:
		score = np.average(item, weights=similarity[k_nearest_rows])
	else:
		score = np.mean(item)
	return score	
		
	
#Experiment 1 - User-User similarity - input data_file,query_file
#Experiment 2 - Movie-Movie similarity - input data_file,query_file,True
def basic_cf(data_file,query_file,output_file,k,cosine=True,weighted=True,movies=False):
	start_time = time.time()
	X = create_data_matrix(data_file)
	if movies:
		X = csr_matrix(X.transpose())
	
	#subtract 3 from each of the non-empty cells
	X.data -=3
	
	#cosine similarity:
	if cosine:
		norm_X = normalize(X)
		X_sim = norm_X.dot(norm_X.transpose())
	else:
		X_sim = X.dot(X.transpose())
	
	Q = read_query(query_file)
	f =open(output_file,'w')
	for query in Q:
		row,col = (query[1],query[0]) if movies else (query[0],query[1])
		knn_score = knn(X,k,row,col,X_sim,weighted,True) 
		knn_score +=3 			#re-add the 3 I substracted earlier
		f.write('%f\n'%knn_score)
	f.close()
	print "Time: ",(time.time()-start_time)
	
#Experiment 3 :PCC-based CF
def pcc_cf(data_file,query_file,output_file,k,cosine=True,weighted=True,movies=False):
	start_time = time.time()
	X = create_data_matrix(data_file)
	if movies:
		X = csr_matrix(X.transpose())
	
	#stdev + mean calculation:
	M = np.zeros(X.shape[0])
	S = np.zeros(X.shape[0])
	for i in range(X.shape[0]):
		if X[i].size:					#non-empty row
			M[i] = np.mean(X[i].data)
			S[i] = math.sqrt(np.var(X[i].data))
			#subtract mean and divide by stdev:
			X.data[X.indptr[i]:X.indptr[i+1]] -=M[i]	#slicing is right exclusive =only row i
			if S[i]>0:
				X.data[X.indptr[i]:X.indptr[i+1]] /=S[i]
	#cosine similarity:
	if cosine:
		norm_X = normalize(X)
		X_sim = norm_X.dot(norm_X.transpose())
	else:
		X_sim = X.dot(X.transpose())
	
	Q = read_query(query_file)
	f =open(output_file,'w')
	
	for query in Q:
		row,col = (query[1],query[0]) if movies else (query[0],query[1])
		knn_score = knn(X,k,row,col,X_sim,weighted,True) #weighted
		#re-add the score after normalization by stdev and mean: (with upper/lower bound)
		knn_score *=S[row]
		knn_score +=M[row]
		if knn_score<1:
			knn_score=1
		if knn_score>5:
			knn_score=5
		f.write('%f\n'%knn_score)
	f.close()
	print "Time: ",(time.time()-start_time)
	

def bpc_cf(data_file,query_file,output_file,k,cosine=True,weighted=True,movies=False):
	start_time = time.time()
	X = create_data_matrix(data_file)
	X.data -=3
	Y = X.transpose(copy=True)
	#print 'beforebpc'
	#X2XC,Y2YC = bipartite_clustering(X,Y,X.shape[0]/3,Y.shape[0]/3,0.01)
	X2XC,Y2YC = bipartite_clustering(X,	1000,500,'cosine',0.1)
	#print 'afterbpc'
	if movies:
		X,Y = Y,X
		X2XC,Y2YC = Y2YC,X2XC
	X_centroids = get_new_centroids(X,X2XC)
	
	if cosine:
		X_norm = normalize(X)
		X_sim = X_norm.dot(normalize(csr_matrix(X_centroids.transpose())))
	else:
		X_sim = X.dot(csr_matrix(X_centroids.transpose()))
	
	Q = read_query(query_file)
	f =open(output_file,'w')
	
	for query in Q:
		row,col = (query[1],query[0]) if movies else (query[0],query[1])
		knn_score = knn(X_centroids,k,X2XC[row],col,X_sim,weighted,True) #weighted
		knn_score+=3
		f.write('%f\n'%knn_score)
	f.close()
	print "Time: ",(time.time()-start_time)
	
def main():
	train_file = "train.csv"
	query_file = "dev.csv"
	output1 = "out1.csv"
	
	pcc_cf(train_file,query_file,"bestout1.csv",10,False,False,False)
	pcc_cf(train_file,query_file,"bestout2.csv",10,False,False,True)
	
	return 0
	
	
if __name__== "__main__":
	main()