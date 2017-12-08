from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from constants import STOPWORDS
from preprocessor import LancasterTokenizer


vectorizer = CountVectorizer(stop_words=STOPWORDS, tokenizer=LancasterTokenizer())
print vectorizer
CountVectorizer(stop_words=STOPWORDS)
test_data = ("Jeffrey Fei walked into a bar. He saw Ben Wu.", "Ben Wu sees him. Ben says that he's sad.")
train_data = ("Jeffrey Fei ran into a bar. Ben Wu sees him.", "He saw Ben Wu. Jeffrey says that he's sad.")
tags = vectorizer.fit_transform(train_data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(tags)
print vectorizer.vocabulary_
print X_train_tfidf

