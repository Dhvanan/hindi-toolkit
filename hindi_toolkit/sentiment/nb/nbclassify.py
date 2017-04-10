import sys
import os

tprob_dict = {}
priors = {}
# fo = open("nboutput.txt","w")


def read_model():
	f = open("../models/nbmodel.txt","r")
	keys = f.readline().split('\t\t')[1:]
	priors_line = f.readline().split('\t')
	priors['1'] = float(priors_line[0])
	priors['-1'] = float(priors_line[1])
	priors['0'] = float(priors_line[2])

	f.readline()
	for line in f:
		values = line.split('\t')
		token = values[0]
		tprob_dict[token] = {}
		tprob_dict[token]['1'] = values[1]
		tprob_dict[token]['-1'] = values[2]
		tprob_dict[token]['0'] = values[3]

	return priors, tprob_dict

def classify(line, priors, tprob_dict):
	prods = {\
	'1':priors['1']\
	,'-1':priors['-1']\
	,'0':priors['0']\
	}
	token_list = [word[0] for word in line]
	for token in token_list:
		if token not in tprob_dict:
			continue
		else:
			for key in ['1','-1','0']:
				prods[key] += float(tprob_dict[token][key])

	if prods['1']>prods['-1'] and prods['1']>prods['0']:
		class1 = '1'
	elif prods['-1']>prods['1'] and prods['-1']>prods['0']:
		class1 = '-1'
	elif prods['0']>prods['-1'] and prods['0']>prods['1']:
		class1 = '0'

	return class1


def predict_sentiment(sentences, priors, tprob_dict):
	#read data
	sentiments = []
	for line in sentences:
		sentiments.append(int(classify(line, priors, tprob_dict)))
	return sentiments

if __name__ == '__main__':
	fname = sys.argv[1]
	main(fname)
