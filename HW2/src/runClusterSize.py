import time
import bpc			#bipartite clustering
import argparse
import sys

def main():
	#optional arguments:
	#argument_parser = argparse.ArgumentParser(description='Bipartite Clustering algorithm.')
	#argument_parser.add_argument('-wc','--wc',type=int,default=100,help='number of word clusters. default=100.')
	#argument_parser.add_argument('-dc','--dc',type=int,default=50,help='number of document clusters. default=100.')
	#argument_parser.add_argument('-test','--test',help="Work with test set.",action='store_true')
	#parsed_arg = argument_parser.parse_args()
	
	test_docVector = "HW2_test.docVectors"
	gold_std = "HW2_dev.gold_standards"
	dev_docVector = "HW2_dev.docVectors"
	word_df = "HW2_dev.df"
	filename = dev_docVector
	
	doc_output = "doc_clusters.out"
	word_output = "word_clusters.out"
	begin_time = time.time()
	for wc in (50,100,150,200):
		for dc in (50,100,150,200):
			begin_time = time.time()
			bpc.main(wc,dc,filename,doc_output,word_output,word_df,0.1)
			output = {}
			output['wc']=wc
			output['dc']=dc
			output['runtime'] = time.time()-begin_time
			print output
			sys.argv = ['eval.py',doc_output,gold_std]
			execfile('eval.py')
			print('')
			print('')
	#output = {}
	#output['runtime'] = time.time()-begin_time
	#print output
	#sys.argv = ['eval.py',doc_output,gold_std]
	#execfile('eval.py')
	#print('')
	#print('')
if __name__== "__main__":
	main()