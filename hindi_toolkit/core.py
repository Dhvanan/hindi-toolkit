# coding: utf-8

from pos.pos_tagger import pos_tag, tokenize
from pos.pos_tagger import load_model as load_pos_model
from ner.crf_ner import tag_ner
from ner.crf_ner import load_model as load_ner_model

import codecs


class Core():

    def __init__(self, sentiment_analyzer='nb'):

        # Load the models
        self.models = {}
        self.models['pos'] = load_pos_model()
        self.models['ner'] = load_ner_model()

    def load(self, doc, process=True):
        self.doc = doc
        self.sentences = []
        self.coref_sentences = []
        self.sentiment = []

        if process == True:
            # self.split_sentence()\
            #     .tokenize()\
            #     .tag_pos()\
            #     .tag_ner()\
            #     .predict_sentiment()

            self.tokenize()\
                .tag_pos()\
                .tag_ner()\
                # .predict_sentiment()

    def split_sentence(self):
        pass
        # self.sentences = split_sentence(self.doc)
        return self

    def tokenize(self):
        self.sentences = tokenize(self.doc)
        # Replace sentence string with list of tokens
        # Format -> [token, <POS_tag>, <NE_tag>]
        # Initialize the <POS_tag>, <NE_tag> to None
        return self

    def tag_pos(self):
        pos_tag(self.sentences, self.models['pos'])
        # self.sentences = tag_pos(self.sentences)
        # Replace <POS_tag> field with prediction
        return self

    def tag_ner(self):
        tag_ner(self.sentences, self.models['ner'])
        return self

    def predict_sentiment(self):
        # self.sentiment = predict_sentiment(self.sentences)
        pass

    def plot_sentiment_curve(self):
        plot_sentiment_curve(self.sentiment)

    def write_to_file(self, file_path):
        outfile = codecs.open(file_path, 'w', 'utf-8')

        for i in range(len(self.sentences)):
            for token in self.sentences[i]:
                outfile.write(token[0] + '\t' + token[1] +
                              '\t' + token[2] + '\n')
            try:
                outfile.write('Sentiment: ' + self.sentiment[i] + '\n\n')
            except:
                pass

        outfile.close()
