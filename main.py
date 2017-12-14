import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from constants import STOPWORDS
from rake import Rake
from preprocessor import LancasterTokenizer, pop_subject_from_document

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

def find_main_concepts(sentences):
    word_ranking = {}
    for sentence in sentences:
        tagged_words = pos_tag(word_tokenize(sentence))
        for tagged_word in tagged_words:
            word = tagged_word[0]
            tag = tagged_word[1]
            if tag == 'NN':
                if word in word_ranking:
                    word_ranking[word] += 1
                else:
                    word_ranking[word] = 1
    return sorted(word_ranking, key=word_ranking.get, reverse=True)[:15]

def contains_main_concepts(sentence, concepts):
    for word in word_tokenize(sentence):
        if word in concepts:
            return True
    return False

def contains_proper_nouns(sentence):
    tagged_words = pos_tag(word_tokenize(sentence))
    for tagged_word in tagged_words:
        if tagged_word[1] == 'NNP':
            return True
    return False

def get_sentence_keyword_score(document, num_sentences):
    rake = Rake()
    keywords = rake.get_keywords(document)
    ranked_keywords = rake.generate_keyword_rank(keywords)
    sufficient_keywords_length = int(math.ceil(len(ranked_keywords) / 3.0))
    sufficient_keywords = ranked_keywords[:sufficient_keywords_length]
    total_keyword_score = 0.0
    # value of a keyword is its relative score value divided by the score of all keywords
    sentence_keyword_score = [0.0] * num_sentences
    for keyword in sufficient_keywords:
        total_keyword_score += keyword['score']
    for keyword in sufficient_keywords:
        sentence_keyword_score[keyword['sentence_num']] += keyword['score'] / total_keyword_score
    return sentence_keyword_score


def main(documents):
    for document in documents:
        title, body = pop_subject_from_document(document)
        sentences = sent_tokenize(body)
        sentence_keyword_score = get_sentence_keyword_score(body, len(sentences))
        sentence_data = []
        word_tokens = tokenize(sentences)
        sentence_title_similarities = find_title_similarity_measure(title, sentences)
        # start processing different attributes of every sentence
        sentence_tf_isf = find_avg_tfidf(transform_tfidf(word_tokens).toarray())
        max_sentence_length = find_max_sentence_length(sentences)
        concepts = find_main_concepts(sentences)
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence) / float(max_sentence_length)
            sentence_pos = (len(sentences) - i) / float(len(sentences))
            keyword_similarity = sentence_keyword_score[i]

            sentence_data.append({
                'sentence': sentence,
                'avg_tf_isf': sentence_tf_isf[i],
                'len_ratio': sentence_length,
                'keyword_similarity': keyword_similarity,
                'pos': sentence_pos,
                'simlarity_to_title': sentence_title_similarities[i],
                'has_main_concepts': contains_main_concepts(sentence, concepts),
                'has_proper_noun': contains_proper_nouns(sentence)
            })
        print(sentence_data)

main(train_data.data[:1])

