from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
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

def find_avg_tfidf(tfisf_array):
    '''
    averages out the tfidf values of every sentence,
    returns a list of averages by sentence position
    '''
    sentence_avg_tfidf = []
    for sentence in tfisf_array:
        total = 0
        for tf_isf_val in sentence:
            total += tf_isf_val
        avg = total / len(sentence)
        sentence_avg_tfidf.append(avg)
    return sentence_avg_tfidf


def find_max_sentence_length(sentences):
    max_length = 0
    for s in sentences:
        if len(s) > max_length:
            max_length = len(s)
    return max_length


def find_title_similarity_measure(title, sentences):
    tfidf_transformer = TfidfVectorizer()
    title_and_sentences = [title] + sentences
    tfidf = tfidf_transformer.fit_transform(title_and_sentences)
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity.A[0]

def main(documents):
    for document in documents:
        title, body = pop_subject_from_document(document)
        paragraphs = [p for p in body.split('\n') if p]
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            sentence_data = []
            word_tokens = tokenize(sentences)
            sentence_title_similarities = find_title_similarity_measure(title, sentences)
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
                    'simlarity_to_title': sentence_title_similarities[i]
                })
            print sentence_data

main([train_data.data[1]])

