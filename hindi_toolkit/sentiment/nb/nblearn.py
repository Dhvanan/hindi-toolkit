from __future__ import division
import sys
import math
from tokenize import *

tfreq_dict = {}
sum_keys = {'1':0,'-1':0,'0':0}
priors = {'1':0,'-1':0,'0':0}

def write_to_file():
	fo = open("../../models/nbmodel.txt","w")
	
	first_line = "Priors\t\t1\t\t-1\t\t0\n"
	fo.write(first_line)
	priors_line = str(math.log(priors['1']/(priors['1']+priors['-1']+priors['0'])))\
	+'\t'+str(math.log(priors['-1']/(priors['1']+priors['-1']+priors['0'])))\
	+'\t'+str(math.log(priors['0']/(priors['1']+priors['-1']+priors['0'])))
	fo.write(priors_line+"\n")
	
	first_line = "Model\t\t1\t\t-1\t\t0\n"
	fo.write(first_line)
	for token in tfreq_dict:
		line = '\t'.join([\
			token,\
			str(tfreq_dict[token]['1']),\
			str(tfreq_dict[token]['-1']),\
			str(tfreq_dict[token]['0']),\
			])
		fo.write(line+"\n")
	fo.close()

def cal_prob():
	for token in tfreq_dict:
		for key in tfreq_dict[token]:
			tfreq_dict[token][key] = math.log(tfreq_dict[token][key]/sum_keys[key])

def smooth():
	for token in tfreq_dict:
		for key in tfreq_dict[token]:
			tfreq_dict[token][key]+=1
			#sum the freq of each class
			sum_keys[key]+=tfreq_dict[token][key]

def read_input(f1_name,f2_name):
	
	#read line by line from file 1 and file 2
	with open(f1_name) as f1, open(f2_name) as f2: 
		for line1, line2 in zip(f1,f2):
			#get the classes the line belongs to
			class1 = line1.strip()
			priors[class1]+=1
			#tokenize each line
			token_list = tokenize(line2)
			#store the tokens in a dictionary
			#update freq of token
			for token in token_list:
				if token not in tfreq_dict:
					tfreq_dict[token] = {'1':0,'-1':0,'0':0}
				tfreq_dict[token][class1]+=1

#def main(f1_name,f2_name):
def main():

	#read input and store in a dictionary
	files = [("../../data/stories/"+str(c)+".txt","../../data/sentiment/stories/"+str(c)+"_sen.txt") for c in [1,2,4]]
	for tup in files:
		read_input(tup[1], tup[0])
	
	#smoothing
	smooth()
	
	#cal prob
	cal_prob()
	
	#write to file
	write_to_file()

if __name__ == '__main__':
	#f1_name,f2_name = sys.argv[1],sys.argv[2]
	#main(f1_name,f2_name)
	main()
