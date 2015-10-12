import sys
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix,lil_matrix
import math

def page_rank(trans_m,n,iter,alpha):
	r_zero = np.ones((n,1),dtype=float)
	r_zero = normalize(r_zero,norm='l1',axis=0,copy=False)
	p_zero = r_zero
	mT = trans_m.transpose()
	pr_lval = np.multiply(mT,(1-alpha))
	pr_rval = np.multiply(p_zero,alpha)
	last_r = r_zero
	for i in range(iter):
		print i,
		if i>0:
			tmp  = np.multiply(pr_lval,last_r)
			r = tmp+pr_rval
			last_r = r
	return r
def main():
	transition_file = "./transition.txt"
	with open(transition_file) as f:
		transition_cells = f.readlines()
	
	rows = []
	cols = []
	data = []
	for single_cell in transition_cells:
		cell_info=single_cell.strip().split(" ")
		row_id = cell_info[0]
		col_id = cell_info[1]
		cell = cell_info[2]
		#print 'row:',row_id,' col:',col_id,' data:',cell	
		rows.append(row_id)
		cols.append(col_id)
		data.append(cell)			#only supposed to be 1's
	max_row = np.amax(np.array(rows).astype(int))+1
	max_col = np.amax(np.array(cols).astype(int))+1
	matrix_size = max(max_row,max_col)	
	raw_link_matrix = csr_matrix((np.array(data).astype(float),(np.array(rows),np.array(cols))),shape=(matrix_size,matrix_size))
	transition_matrix = normalize(raw_link_matrix,norm='l1',axis=1,copy=False)
	page_rank(transition_matrix,matrix_size,10,0.1)
	
	
if __name__== "__main__":
	main()