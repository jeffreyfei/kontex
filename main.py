from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import sent_tokenize

from constants import STOPWORDS
from preprocessor import LancasterTokenizer

# TODO: use actual data i.e. news articles
test_data = ("Jeffrey Fei walked into a bar. He saw Ben Wu in the bar.", "Ben Wu sees him. Ben says that he's sad.")
train_data = ("Jeffrey Fei ran into a bar. Ben Wu sees him", "He saw Ben Wu. Jeffrey says that he's sad.", "Hello there. I am a dog")

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

def find_normalized_sentence_lengths(sentences):
    max_length = 0
    for s in sentences:
        if len(s) > max_length:
            max_length = len(s)
    sentence_lengths = []
    for s in sentences:
        len_ratio = len(s) / float(max_length)
        sentence_lengths.append(len_ratio)
    return sentence_lengths
    
def main(documents):
    for document in documents:
        sentences = sent_tokenize(document)
        sentence_data = []
        word_tokens = tokenize(sentences)

        # start processing different attributes of every sentence
        sentence_tf_isf = find_avg_tfidf(transform_tfidf(word_tokens).toarray())
        sentence_lengths = find_normalized_sentence_lengths(sentences)
        
        for i in xrange(len(sentences)):
            sentence_data.append({
                'sentence': sentences[i],
                'avg_tf_isf': sentence_tf_isf[i],
                'len_ratio': sentence_lengths[i],
            })
        print sentence_data

main(train_data)

