# coding: utf-8

from pos.pos_tagger import pos_tag
from ner.crf_ner import tag_ner
from ner.crf_ner import load_model as load_ner_model


class Core():

    def __init__(self):
        return
        # Load the models
        self.models = {}
        self.models['ner'] = load_ner_model()

    def load(self, doc, process=True):
        self.doc = doc
        self.sentences = []
        self.coref_sentences = []
        self.sentiment = []

        if process == True:
            self.split_sentence()
            self.tokenize()
            self.tag_pos()
            self.tag_ner()

            self.predict_sentiment()

    def split_sentence(self):
        print('Split document into sentences')
        # self.sentences = split_sentence(self.doc)

    def tokenize(self):
        print('Tokenize sentences')
        # self.sentences = tokenize(self.sentences)
        # Replace sentence string with list of tokens
        # Format -> [token, <POS_tag>, <NE_tag>]
        # Initialize the <POS_tag>, <NE_tag> to None

    def tag_pos(self, text):
        return pos_tag(text)
        # self.sentences = tag_pos(self.sentences)
        # Replace <POS_tag> field with prediction

    def tag_ner(self):
        tag_ner(self.sentences, self.models['ner'])

    def predict_sentiment(self):
        print 'Sentiment Analysis'
        # self.sentiment = predict_sentiment(self.sentences)
