# coding: utf-8

from ner.crf_ner import tag_ner
from pos.pos_tagger import pos_tag

class Core():

    def __init__(self):
        return
        # Load the models

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
        self.sentences = tag_net(self.sentences)

    def predict_sentiment(self):
    	self.sentiment = predict_sentiment(self.sentences)