# coding: utf-8

from ner.crf_ner import tag_ner


class Core():

    def __init__(self):
        self.doc = None
        self.sentences = []
        # contains list of sentences, where each sentence is a list of tokens
        # and pos tags

        self.coref_sentences = []
        self.sentiment = []
        # contains predicted sentiment score for each sentence

        # Load the models

    def process(self, doc):
        self.doc = doc
        self.sentences = []
        self.coref_sentences = []
        self.sentiment = []

        self.split_sentence()
        self.tokenize()
        self.tag_pos()
        self.tag_ner()

    def split_sentence(self):
        print 'Split document into sentences'
        # self.sentences = split_sentence(self.doc)

    def tokenize(self):
        print 'Tokenize sentences'
        # self.sentences = tokenize(self.sentences)
        # Replace sentence string with list of tokens
        # Format -> [token, <POS_tag>, <NE_tag>]
        # Initialize the <POS_tag>, <NE_tag> to None

    def tag_pos(self):
        print 'POS Tagger'
        # self.sentences = tag_pos(self.sentences)
        # Replace <POS_tag> field with prediction

    def tag_ner(self):
        self.sentences = tag_net(self.sentences)
