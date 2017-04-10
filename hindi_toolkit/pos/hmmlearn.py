#!/usr/bin/env python
# -*- coding: UTF-8 -*-


"""
Author : Dhvanan Shah
HMM POS Tagging for Catalan documents
HMM Training

TRAINING
Tagged Format : 
Sentence on each line
word/tag semission_parated by space
"""



import math
import sys
import json
import os
from tokenizer import tokenize


def create_model(directory):
	emission_p = {}
	transition_p = {}
	initial_p = {}
	tag_counts = {}
	tags = set()

	files = os.listdir(directory)
	for file_name in files:
		with open(directory+file_name,'r') as f:
			data = f.read().strip().split('\n\n')



		for sentence in data:
			sentence = sentence.strip().split(' ')
			print(sentence)
			prev_tag = 0
			for token in sentence:
				print(token)
				word,tag = token.split('/')
				
				if(tag not in tags):
					tag_counts[tag] = 1
					tags.add(tag)

				else:
					tag_counts[tag] += 1	


				#Add to the EMISSION PROB
				if(word not in emission_p):
					emission_p[word] = {}
				if(tag not in emission_p[word]):
					emission_p[word][tag] = 1
				else:
					emission_p[word][tag] += 1


				#Add to the INITIAL PROB
				if(prev_tag == 0):
					if(tag not in initial_p):
						initial_p[tag] = 1
					else:
						initial_p[tag] += 1

				#Add to the TRANSITION PROB
				if(prev_tag!=0):
					if(prev_tag not in transition_p):
						transition_p[prev_tag] = {}
					if(tag not in transition_p[prev_tag]):
						transition_p[prev_tag][tag] = 1
					else:
						transition_p[prev_tag][tag] += 1

				prev_tag = tag



	transition_p = add_one_smoothing_transition(transition_p,tags)
	initial_p = add_one_smoothing_initial(initial_p,tags)
	emission_p = normalize(emission_p,tag_counts)


	return (initial_p,transition_p,emission_p)


def add_one_smoothing_transition(transition_p,tags):
	for tag1 in transition_p:
		denominator = sum(transition_p[tag1].values()) + len(tags)
		for tag2 in tags:
			if(tag2 not in transition_p[tag1]):
				transition_p[tag1][tag2] = 1
			else:
				transition_p[tag1][tag2] += 1
			transition_p[tag1][tag2] = math.log(transition_p[tag1][tag2] / float(denominator))

	return transition_p	

def add_one_smoothing_initial(initial_p,tags):
	denominator = sum(initial_p.values())+len(tags)

	for tag in tags:
		if(tag not in initial_p):
			initial_p[tag] = 1
		else:
			initial_p[tag] += 1

		initial_p[tag] = math.log(initial_p[tag] / float(denominator))

	return initial_p


def normalize(emission_p,tag_count):
	for word in emission_p:
		for tag in emission_p[word]:
			emission_p[word][tag] = math.log((emission_p[word][tag]) / float(tag_count[tag]))
	return emission_p

def write_model(initial_p,transition_p,emission_p):
	model = {"initial_p" : initial_p , "transition_p" : transition_p , "emission_p" : emission_p}
	with open('hmmmodel.txt','w') as f:
		f.write(json.dumps(model))

if(__name__ == '__main__'):
	file_name = "../Dataset/Training Data/"
	initial_p,transition_p,emission_p = create_model(file_name)
	write_model(initial_p,transition_p,emission_p)