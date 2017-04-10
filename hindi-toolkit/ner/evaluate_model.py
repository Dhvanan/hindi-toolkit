# coding: utf-8

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import json
import codecs
import random

from crf_ner import get_features, get_X_y, train_model
# config
ENCODING = 'utf-8'
FILE_PATH = '../data/ner/train.json'
SEED = 7
SPLIT = 0.20


def load():
    json_data = json.load(codecs.open(
        FILE_PATH, 'r', 'utf-8'), encoding=ENCODING)
    return json_data['sentences']


def shuffle(sentences):
    random.seed(SEED)
    random.shuffle(sentences)


def split(sentences):
    split = int(len(sentences) * SPLIT)
    return {'train': sentences[split:], 'test': sentences[:split]}


def evaluate_model(test, crf):
    X_test, y_test = get_X_y(test)

    labels = list(crf.classes_)
    labels.remove('O')

    y_pred = crf.predict(X_test)

    # F1 score
    f1_score = metrics.flat_f1_score(y_test, y_pred,
                                     average='weighted', labels=labels)
    print 'F1 score : ', f1_score

    print 'NE label wise analysis'
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    print metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    )


if __name__ == '__main__':

    sentences = load()

    ne_count = {}
    for sentence in sentences:
        for word in sentence:
            ne_count[word[len(word) - 1]
                     ] = ne_count.get(word[(len(word) - 1)], 0) + 1

    output = ''
    # Analysis of the NE types
    print 'Analysis of the NE types in the entire Data'
    for ne_type in sorted(ne_count, key=ne_count.get, reverse=True):
        print ne_type, ne_count[ne_type]

    shuffle(sentences)

    data = split(sentences)
    train, test = data['train'], data['test']

    # Analysis of the dataset
    print 'Total # sentences : \t', len(sentences)
    print '# sentences in train: \t', len(train)
    print '# sentences in test: \t', len(test)

    # Sample feature extraction
    temp_X, temp_y = get_X_y([train[0]])
    print '\n\n-----------\n'
    print 'Sample feature extraction'
    print temp_X[0][0], temp_y[0][0]

    crf = train_model(train)

    print '\n\n-----------\n'
    evaluate_model(test, crf)
