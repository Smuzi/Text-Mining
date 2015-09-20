import time
import bpc			#bipartite clustering
import argparse
import sys

def main():
	#optional arguments:
	argument_parser = argparse.ArgumentParser(description='Bipartite Clustering algorithm.')
	argument_parser.add_argument('-wc','--wc',type=int,default=100,help='number of word clusters. default=100.')
	argument_parser.add_argument('-dc','--dc',type=int,default=50,help='number of document clusters. default=100.')
	argument_parser.add_argument('-test','--test',help="Work with test set.",action='store_true')
	parsed_arg = argument_parser.parse_args()
	
	test_docVector = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_test.docVectors"
	gold_std = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_dev.gold_standards"
	dev_docVector = "C:\Smuzi\CMU\ML Txt mining\HW\HW2\data\HW2_dev.docVectors"
	if parsed_arg.test:
		filename = test_docVector
	else:
		filename = dev_docVector
	
	doc_output = "doc_clusters.out"
	word_output = "word_clusters.out"
	begin_time = time.time()
	bpc.main(parsed_arg.wc,parsed_arg.dc,filename,doc_output,word_output)
	
	output = {}
	output['runtime'] = time.time()-begin_time
	print output
	sys.argv = ['eval.py',doc_output,dev_docVector]
	execfile('eval.py')

if __name__== "__main__":
	main()