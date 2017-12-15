import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

class LancasterTokenizer(object):
    def __init__(self):
        self.ls = PorterStemmer()
    def __call__(self, doc):
        return [self.ls.stem(word) for word in word_tokenize(doc)]

def pop_subject_from_document(document):
    split_document = document.split("\n")
    header = split_document[1]
    body_begin = split_document.index('') + 1
    body = ' '.join(split_document[body_begin:])
    title_begin = len("Subject: ")
    title = header[title_begin:]
    return title, body
