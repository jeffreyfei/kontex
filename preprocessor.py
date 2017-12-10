import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from constants import STOPWORDS

class LancasterTokenizer(object):
    def __init__(self):
        self.ls = PorterStemmer()
    def __call__(self, doc):
        return [self.ls.stem(word) for word in word_tokenize(doc)]

def pop_subject_from_document(document):
    split_document = document.split("\n")
    header = split_document[1]
    body = ' '.join(split_document[9:])
    title_begin = len("Subject: ")
    title = header[title_begin:]
    return title, body
