import nltk
from nltk.tokenize import word_tokenize
from nltk.collocations import *
from nltk.stem.porter import *
from constants import STOPWORDS

class LancasterTokenizer(object):
    def __init__(self):
        self.ls = PorterStemmer()
    def __call__(self, doc):
        return [self.ls.stem(word) for word in word_tokenize(doc)]