import codecs
import sys

sys.path.append('../')
from core import Core as HindiToolkit


PATH_TO_DOC = '../data/stories/'

if __name__ == '__main__':

    # Initialize the hindi-toolkit object
    htk = HindiToolkit()

    # Read the document
    doc = codecs.open(PATH_TO_DOC + '1.txt', 'r', 'utf-8').read()

    # Load the document into the extraction pipeline
    htk.load(doc)

    htk.write_to_file('output.txt')

    htk.plot_sentiment_curve()
