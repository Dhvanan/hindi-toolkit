# coding: utf-8

import json
import codecs
import sklearn_crfsuite
from sklearn.externals import joblib

PATH_TO_MODELS = '../models/'
PATH_TO_TRAIN = '../data/ner/train.json'
ENCODING = 'utf-8'

def get_features(sentence, i):
    word = sentence[i]

    # general word features
    features = {
        # 'bias': 1.0,
        'word': word[0],
        'word[-3:]': word[0][-3:],
        'word[-2:]': word[0][-2:],
        'word.is_digit': word[0].isdigit(),
        'pos_tag': word[1],
        'pos_tag[:2]': word[1][:2]
    }

    # if not first word
    if i > 0:
        prev_word = sentence[i - 1]
        features.update({
            '-1:word': prev_word[0],
            '-1:pos_tag:': prev_word[1],
            '-1:pos_tag[:2]': prev_word[1][:2],
            '-1:is_digit': prev_word[0].isdigit()
        })

    else:
        features.update({
            'first_word': True
        })

    # if not last word
    if i < len(sentence) - 1:
        next_word = sentence[i + 1]
        features.update({
            '+1:word': next_word[0],
            '+1:pos_tag:': next_word[1],
            '+1:pos_tag[:2]': next_word[1][:2],
            '+1:is_digit': next_word[0].isdigit()
        })

    else:
        features.update({
            'last_word': True
        })

    return features


def get_X_y(sentences):
    X = get_X(sentences)
    y = []

    for sentence in sentences:
        y.append([word[len(word) - 1] for word in sentence])

    return X, y


def get_X(sentences):
    X = []
    for sentence in sentences:
        X.append([get_features(sentence, i)
                  for i in range(len(sentence))])
    return X


def train_model(train):
    X_train, y_train = get_X_y(train)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    return crf


def save_model(train):
    crf = train_model(train)
    joblib.dump(crf, PATH_TO_MODELS + 'crf_ner.pkl')


def load_model():
    crf = joblib.load(PATH_TO_MODELS + 'crf_ner.pkl')
    return crf


def tag_ner(sentences, crf):
    X = get_X(sentences)

    y = crf.predict(X)

    for i in range(len(sentences)):
        for j in range((len(sentences[i]))):
            sentences[i][j][2] = y[i][j]


def load():
    json_data = json.load(codecs.open(
        PATH_TO_TRAIN, 'r', 'utf-8'), encoding=ENCODING)
    return json_data['sentences']


if __name__ == '__main__':

    sentences = load()

    save_model(sentences)
