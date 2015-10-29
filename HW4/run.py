import subprocess
from cf import basic_cf,pcc_cf,bpc_cf

print "Prediction of ratings on test dataset:"
print "Movie-biased PCC-based CF with dot product similarity and mean rating and k=10"
cosine = False
weighted = False
pcc_cf("train.csv","test.csv","prediction.csv",10,cosine,weighted,True)
print "prediction output in file: prediction.csv"
print "-------------------"					
counter = 0
for experiment in (1,2,3,4):
	for cosine in (True,False):
		for k in (10,100,500):
			for weighted in (True,False):
				if weighted:
					rating = 'weighted'
				else:
					rating = 'mean'
				if cosine:
					metric = 'cosine'
				else:
					metric = 'dot product'
				print "experiment "+str(experiment)+" k="+str(k)+" "+rating+" "+metric
				if experiment==1:
					basic_cf("train.csv","dev.csv","out%i.csv"%counter,k,cosine,weighted,False)
				if experiment==2:
					basic_cf("train.csv","dev.csv","out%i.csv"%counter,k,cosine,weighted,True)
				if experiment==3:
					pcc_cf("train.csv","dev.csv","out%i.csv"%counter,k,cosine,weighted,False)
				if experiment==4:
					bpc_cf("train.csv","dev.csv","out%i.csv"%counter,k,cosine,weighted,False)
				s = "python eval_rmse.py dev.golden out%i.csv"%counter
				print subprocess.Popen(s.split(),stdout=subprocess.PIPE).communicate()[0]
				counter +=1
