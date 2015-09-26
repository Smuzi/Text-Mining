import time
import bpc			#bipartite clustering
import argparse
import sys

def main():
	test_docVector = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_test.docVectors"
	gold_std = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_dev.gold_standards"
	dev_docVector = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_dev.docVectors"
	word_df = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_dev.df"
	filename = dev_docVector
	
	doc_output = "doc_clusters.out"
	word_output = "word_clusters.out"
	
	for criteria in (1,0.1,0.01,0.001):
		begin_time = time.time()
		bpc.main(150,75,filename,doc_output,word_output,word_df,criteria)
		output={}
		output['criteria']= criteria
		output['runtime'] = time.time()-begin_time
		print output
		sys.argv = ['eval.py',doc_output,gold_std]
		execfile('eval.py')
		print('')
		print('')
if __name__== "__main__":
	main()