from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import sent_tokenize

from constants import STOPWORDS
from preprocessor import LancasterTokenizer, pop_subject_from_document
from sklearn.datasets import fetch_20newsgroups
train_data = fetch_20newsgroups(subset='train', remove=('footers', 'quotes'))


def tokenize(sentence):
    vectorizer = CountVectorizer(stop_words=STOPWORDS, tokenizer=LancasterTokenizer())
    tags = vectorizer.fit_transform(sentence)
    return tags

def transform_tfidf(words):
    tfidf_transformer = TfidfTransformer()
    return tfidf_transformer.fit_transform(words)

def find_avg_tfidf(tfidf_array):
    '''
    averages out the tfidf values of every sentence,
    returns a list of averages by sentence position
    '''
    sentence_avg_tfidf = []
    for sentence in tfidf_array:
        total = 0
        for tf_idf_val in sentence:
            total += tf_idf_val
        avg = total / len(sentence)
        sentence_avg_tfidf.append(avg)
    return sentence_avg_tfidf


def find_max_sentence_length(sentences):
    max_length = 0
    for s in sentences:
        if len(s) > max_length:
            max_length = len(s)
    return max_length

def main(documents):
    for document in documents:
        title, body = pop_subject_from_document(document)
        print title
        paragraphs = [p for p in body.split('\n') if p]
        for paragraph in paragraphs:
            print paragraph
            sentences = sent_tokenize(paragraph)
            sentence_data = []
            word_tokens = tokenize(sentences)

            # start processing different attributes of every sentence
            sentence_tf_isf = find_avg_tfidf(transform_tfidf(word_tokens).toarray())
            max_sentence_length = find_max_sentence_length(sentences)
            for i, sentence in enumerate(sentences):
                sentence_length = len(sentence) / float(max_sentence_length)
                sentence_pos = (len(sentences) - i) / float(len(sentences))

                sentence_data.append({
                    'sentence': sentence,
                    'avg_tf_isf': sentence_tf_isf[i],
                    'len_ratio': sentence_length,
                    'pos': sentence_pos,
                })
            print sentence_data

main([train_data.data[1]])

