import argparse
import sys
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import normalize
import time
import math

def corpus_exploration(data_file):
	X = create_data_matrix(data_file)
	X_t = csr_matrix(X.transpose())
	
	sim_4321_cosine = normalize(X)
	sim_4321_cosine = sim_4321_cosine.dot(sim_4321_cosine[4321].transpose()).todense().A1
	sim_4321_cosine[4321]=0
	
	sim_4321_dot = X.dot(X[4321].transpose()).todense().A1
	sim_4321_dot[4321]=0
	
	sim_3_cosine = normalize(X_t)
	sim_3_cosine = sim_3_cosine.dot(sim_3_cosine[3].transpose()).todense().A1
	sim_3_cosine[3]=0
	
	sim_3_dot = X_t.dot(X_t[3].transpose()).todense().A1
	sim_3_dot[3]=0
	
	print 'Total movies:', X.shape[1]
	print 'Total users:', X.shape[0]
	print 'Rate 1 count:', np.count_nonzero(X.data==1)
	print 'Rate 3 count:', np.count_nonzero(X.data==3)
	print 'Rate 5 count:', np.count_nonzero(X.data==5)
	print 'Average movie rating:',np.mean(X.data)
	print '--------------------'
	print 'User 4321'
	print 'Number of ratings:',X[4321].data.shape[0]
	print 'Number of 1 ratings:',np.count_nonzero(X[4321].data==1)
	print 'Number of 3 ratings:',np.count_nonzero(X[4321].data==3)
	print 'Number of 5 ratings:',np.count_nonzero(X[4321].data==5)
	print 'Average rating:',np.mean(X[4321].data)
	print '--------------------'
	print 'Movie 3'
	print 'Number of ratings:',X_t[3].data.shape[0]
	print 'Number of 1 ratings:',np.count_nonzero(X_t[3].data==1)
	print 'Number of 3 ratings:',np.count_nonzero(X_t[3].data==3)
	print 'Number of 5 ratings:',np.count_nonzero(X_t[3].data==5)
	print 'Average rating:',np.mean(X_t[3].data)
	print '--------------------'
	print 'Nearest neighbors'
	print '5 NN of user 4321 - dot product',np.argpartition(sim_4321_dot,sim_4321_dot.size-5)[-5:]
	print '5 NN of user 4321 - cosine',np.argpartition(sim_4321_cosine,sim_4321_cosine.size-5)[-5:]
	print '5 NN of movie 3 - dot product',np.argpartition(sim_3_dot,sim_3_dot.size-5)[-5:]
	print '5 NN of movie 3 - cosine',np.argpartition(sim_3_cosine,sim_3_cosine.size-5)[-5:]

	

def create_data_matrix(filename):
	#training data format: "MovieID","UserID","Rating","RatingDate"
	with open(filename,'r') as f:
		split_rows = map(lambda line:line.split(','),f.readlines())
	rating_data = map(lambda x:int(x[2]),split_rows)
	user_movie_pairs = [map(lambda x:int(x[1]),split_rows),map(lambda y:int(y[0]),split_rows)]
	num_of_users = max(user_movie_pairs[0])+1
	num_of_movies = max(user_movie_pairs[1])+1
	return csr_matrix((rating_data,user_movie_pairs),shape=(num_of_users,num_of_movies),dtype=float)
	
	
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
				
def main():
	#parser = argparse.ArgumentParser(description='Netflix Collaborative Filtering')
	train_file = "C:\\Users\\Uzi\\Documents\\GitHub\\Text-Mining\\HW4\\data\\train.csv"
	query_file = "C:\\Users\\Uzi\\Documents\\GitHub\\Text-Mining\\HW4\\data\\dev.csv"
	output1 = "C:\\Users\\Uzi\\Documents\\GitHub\\Text-Mining\\HW4\\out1.csv"
	output2 = "C:\\Users\\Uzi\\Documents\\GitHub\\Text-Mining\\HW4\\out2.csv"
	#basic_cf(train_file,query_file,output1,10)
	#pcc_cf(train_file,query_file,output2,10,True)
	corpus_exploration(train_file)
	return 0
if __name__== "__main__":
	main()