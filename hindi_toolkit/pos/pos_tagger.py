#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Author : Dhvanan Shah
HMM POS Tagging for Hindi documents
HMM Decoding
Viterbi Algorithm
"""


import re
import sys
import json
import numpy as np
import copy
import operator


PATH_TO_MODEL = '../models/'


def viterbi(observations, tags, initial_p, transition_p, emission_p):
    t1 = []
    t2 = []

    for i in observations:
        t1.append({})
        t2.append({})

    # Initial Word - UNKNOWN Word Handling
    if(observations[0] not in emission_p):

        # Get Most Likely Initial Tag
        initial_tag, initial_value = sorted(
            initial_p.items(), key=operator.itemgetter(1), reverse=True)[0]
        t1[0][initial_tag] = initial_value

    # Initial Word - KNOWN Word Handling
    else:
        for t in emission_p[observations[0]]:
            t1[0][t] = initial_p[t] + emission_p[observations[0]][t]
            t2[0][t] = 0

    # Tagging the rest of the sentence
    for i in range(1, len(observations)):

        # UNKNOWN Word
        if(observations[i] not in emission_p):

            # Only calculate transitions for tags from the previous word.
            for t in t1[i - 1]:

                # Get the most likely transition for the prev tag and add the
                # cumulative values.
                tag, value = sorted(transition_p[t].items(
                ), key=operator.itemgetter(1), reverse=True)[0]
                t1[i][tag] = value + t1[i - 1][t]
                t2[i][tag] = t

        # KNOWN Word
        else:
            # Iterate over all the tags for the given word
            for cur_tag in emission_p[observations[i]]:
                most_likely_tag = ''
                max_value = -sys.maxsize

                # Calculate most likely previsous tag for cur_tag from
                # prev_tags (Including transitions and emissions)
                for prev_tag in t1[i - 1]:
                    calculated_value = transition_p[prev_tag][
                        cur_tag] + emission_p[observations[i]][cur_tag] + t1[i - 1][prev_tag]
                    if((calculated_value) > max_value):
                        most_likely_tag = prev_tag
                        max_value = calculated_value

                t1[i][cur_tag] = max_value
                t2[i][cur_tag] = most_likely_tag

    final_tags = []
    final_tag, final_max_value = sorted(
        t1[-1].items(), key=operator.itemgetter(1), reverse=True)[0]
    final_tags.append(final_tag)

    cur_tag = final_tag
    for i in range(len(observations))[:0:-1]:
        prev_tag = t2[i][cur_tag]
        final_tags.insert(0, prev_tag)
        cur_tag = prev_tag

    return final_tags


def tokenize(text):

    punctuation = ["?", "'", ",", "!", "\"", "-", ";", ":", "."]

    #text = text.replace("।"," . ")
    text = text.replace("?", " ? ")
    text = text.replace("'", " ' ")
    text = text.replace(",", " , ")
    text = text.replace("!", " ! ")
    text = text.replace("\\", " \ ")
    text = text.replace("-", " - ")
    text = text.replace(";", " ; ")
    text = text.replace(":", " : ")
    text = text.replace(".", " . ")

    text = re.sub(' +', ' ', text)

    sentences = text.split('।')
    tokenized_text = []
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = sentence.replace('।', ' .')
        tokenized_text.append([[x, None, None] for x in sentence.split(' ')])

    return tokenized_text


def tag_sentence(tags, sentence):

    # tagged_sentence = []

    # for i in range(len(sentence)):
    #     tagged_sentence.append((sentence[i], tags[i]))

    # return tagged_sentence

    for i in range(len(sentence)):
        sentence[i][1] = tags[i]


def write_output(data):
    with open('hmmoutput.txt', 'w') as f:
        for sentence in data:
            f.write(str(sentence) + '\n')


def pos_tag(sentences, model):
    initial_p = model['initial_p']
    transition_p = model['transition_p']
    emission_p = model['emission_p']
    tags = initial_p.keys()

 #    data = text
 #    """
    # file_name = "../Dataset/Test Data/test.txt"
    # with open(file_name,'r') as f:
    # 	data = f.read().strip()
    # """
 #    tokenized_data = tokenize(data)

    # tagged_data = []
    for sentence in sentences:
        tokens = [token[0] for token in sentence]
        tags = viterbi(tokens, tags, initial_p, transition_p, emission_p)
        tag_sentence(tags, sentence)
        # tagged_data.append(tagged_sentence)

    # write_output(tagged_data)

    # return tagged_data


def load_model():
    with open(PATH_TO_MODEL + 'hmmmodel.txt', 'r') as f1:
        file_content = f1.read()
        model = json.loads(file_content)

        return model
