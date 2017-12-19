import math
import sys  

from copy import deepcopy

from sklearn import metrics
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from rake import Rake
from preprocessor import LancasterTokenizer, fetch_datasets, word_filter_tokenize
from pyteaser import Summarize
# TODO: combine results of multiple summarizers to improve accuracy
# from gensim.summarization import summarize as gensim_summarize


# ensure nltk doesn't crash for non-ascii tokens that pass through
reload(sys)
sys.setdefaultencoding('utf8')

train_data = fetch_datasets()

def tokenize(sentence):
    vectorizer = CountVectorizer(
        stop_words=stopwords.words('english'),
        tokenizer=LancasterTokenizer()
    )
    tags = vectorizer.fit_transform(sentence)
    return tags

def transform_tfidf(words):
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
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
    cos_sim = []
    for sentence in sentences:
        cos_sim.append(get_cos_sim(title, sentence))
    return cos_sim

def find_main_concepts(sentences):
    word_ranking = {}
    for sentence in sentences:
        tagged_words = pos_tag(word_filter_tokenize(sentence))
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
    for word in word_filter_tokenize(sentence):
        if word in concepts:
            return 1.0
    return 0.0

def contains_proper_nouns(sentence):
    tagged_words = pos_tag(word_filter_tokenize(sentence))
    for tagged_word in tagged_words:
        if tagged_word[1] == 'NNP':
            return 1.0
    return 0.0

def get_sentence_keyword_score(document, num_sentences):
    rake = Rake()
    keywords = rake.get_keywords(document)
    ranked_keywords = rake.generate_keyword_rank(keywords)
    sufficient_keywords_length = int(math.ceil(len(ranked_keywords) / 4.0))
    sufficient_keywords = ranked_keywords[:sufficient_keywords_length]
    total_keyword_score = 0.0
    # value of a keyword is its relative score value divided by the score of all keywords
    sentence_keyword_score = [0.0] * num_sentences
    for keyword in sufficient_keywords:
        total_keyword_score += keyword['score']
    for keyword in sufficient_keywords:
        sentence_keyword_score[keyword['sentence_num']] += keyword['score'] / total_keyword_score
    return sentence_keyword_score

def get_vector_magnitude(vector):
    sqlSum = 0
    for i in vector:
        sqlSum += i ** 2
    return math.sqrt(sqlSum)

def get_dot_product(v1, v2):
    dp = 0
    for i, val in enumerate(v1):
        dp += val * v2[i]
    return dp

def get_cos_sim(s1, s2):
    v1 = {}
    sent_tok1 = word_filter_tokenize(s1)
    sent_tok2 = word_filter_tokenize(s2)
    for word in sent_tok1 + sent_tok2:
        v1[word] = 0
    v2 = deepcopy(v1)
    for word in sent_tok1:
        v1[word] += 1
    for word in sent_tok2:
        v2[word] += 1
    m1 = get_vector_magnitude(list(v1.values()))
    m2 = get_vector_magnitude(list(v2.values()))
    dp = get_dot_product(list(v1.values()), list(v2.values()))
    if m1 == 0 or m2 == 0:
        return 0
    else:
        return dp / (m1 * m2)

def get_sent_to_doc_raw_sums(sentences):
    rawSums = []
    for sentence in sentences:
        rawSum = 0
        for other_sent in sentences:
            if sentence != other_sent:
                rawSum += get_cos_sim(sentence, other_sent)
        rawSums.append(rawSum)
    return rawSums

def get_summarized_sentences(base_summary, sentences):
    good_sentences = []
    normalized_summary = [sent.lstrip() for sent in base_summary]
    for sent in sentences:
        good_sentences.append(sent.lstrip() in base_summary)

    return good_sentences

def compute_sentence_data(documents, mode="train"):
    sentence_data = []
    chosen_sentences = []
    for doc_num, document in enumerate(documents):
        if mode == "train":
            print "training through document {} out of {}".format(doc_num+1, len(documents))
        title = document['header']
        body = document['body']
        sentences = sent_tokenize(body)
        base_summary = Summarize(title, body)
        chosen_sentences = chosen_sentences + get_summarized_sentences(base_summary, sentences)

        sentence_keyword_score = get_sentence_keyword_score(body, len(sentences))
        word_tokens = tokenize(sentences)
        sentence_title_similarities = find_title_similarity_measure(title, sentences)
        rawSums = get_sent_to_doc_raw_sums(sentences)
        maxRawSum = max(rawSums)
        # start processing different attributes of every sentence
        sentence_tf_isf = find_avg_tfidf(transform_tfidf(word_tokens).toarray())
        max_sentence_length = find_max_sentence_length(sentences)
        concepts = find_main_concepts(sentences)
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence) / float(max_sentence_length)
            sentence_pos = (len(sentences) - i) / float(len(sentences))
            keyword_similarity = sentence_keyword_score[i]
            sent_sent_cohesion = rawSums[i] / maxRawSum

            sentence_properties = [
                sentence_tf_isf[i],
                sentence_length,
                keyword_similarity,
                sentence_pos,
                sentence_title_similarities[i],
                contains_main_concepts(sentence, concepts),
                contains_proper_nouns(sentence),
                sent_sent_cohesion
            ]
            sentence_data.append(sentence_properties)


    return sentence_data, chosen_sentences

    
def main(documents):
    clf = ComplementNB()
    training_data, training_results = compute_sentence_data(documents[:10])
    print "classifying training data"
    clf=clf.fit(training_data, training_results)
    print "finished training data, computing test data"
    test_data, test_results = compute_sentence_data(documents[:3], mode="test")
    predicted = clf.predict(test_data)
    print(predicted)
    print(test_results)
    print(metrics.classification_report(test_results, predicted,target_names=["not picked", "picked"]))

main(train_data)
