import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import csv

class LancasterTokenizer(object):
    def __init__(self):
        self.ls = PorterStemmer()
    def __call__(self, doc):
        return [self.ls.stem(word) for word in word_tokenize(doc)]


def fetch_datasets():
    data = []
    csv.field_size_limit(500 * 1024 * 1024)
    i = 0
    with open('training_data/articles1.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            i += 1
            sanitized_body = row['content'].decode('utf-8').encode('ascii', errors='ignore')
            data.append({
                'header': row['title'],
                'body': sanitized_body,
            })
            if i > 100:
                break
    return data
