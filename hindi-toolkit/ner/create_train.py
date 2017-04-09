# coding: utf-8

import codecs
import json

input_file = codecs.open('../data/ner/train', 'r', 'utf-8')

sentences = []

sentence = []
for line in input_file:
    if line == '\n':
        sentences.append(sentence)
        sentence = []

    else:
        row = line.split('\t')
        word = []
        for entry in row:
            word.append(entry)
        sentence.append(word)

input_file.close()

output_file = codecs.open('../data/ner/train.json', 'w', 'utf-8')
json.dump({'sentences': sentences}, output_file,
          sort_keys=True, indent=4, encoding='utf-8')
