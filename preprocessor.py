import nltk
from nltk.tokenize import word_tokenize
from nltk.collocations import *
from nltk.stem.lancaster import LancasterStemmer

from constants import STOPWORDS

sentence = "Jeffrey Fei walked into a bar. He saw Ben Wu. Ben Wu sees him. Ben says that he's sad."

def remove_stopwords(text):
    '''
    Removes all meaningless words in the text
    '''
    tokens = nltk.word_tokenize(text)
    filtered_words = [word for word in tokens if word not in STOPWORDS]
    return filtered_words

def stemmer(text):
    st = LancasterStemmer()
    stemmedWords = [st.stem(word) for word in text]
    return stemmedWords


words = stemmer(remove_stopwords(sentence))
print words