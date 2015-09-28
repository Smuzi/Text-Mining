import time
import bpc			#bipartite clustering
import argparse
import sys

def main():	
	test_docVector = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_test.docVectors"
	gold_std = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_dev.gold_standards"
	dev_docVector = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_dev.docVectors"
	word_df = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_dev.df"
	filename = test_docVector
	
	doc_output = "usmadja-test-clusters.txt"
	word_output = "test-word_clusters.out"
	begin_time = time.time()
	criteria = 0.01
	bpc.main(150,75,filename,doc_output,word_output,word_df,criteria)
	
	output = {}
	output['runtime'] = time.time()-begin_time
	print output
if __name__== "__main__":
	main()