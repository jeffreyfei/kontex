from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.corpus import stopwords

class Rake(object):

    def __init__(self, stop_words=stopwords.words('english')):
        self.stopwords = stop_words

    def get_keywords(self, document):
        '''
        groups keywords which are separated by stop words and punctuation
        '''
        sentences = sent_tokenize(document)
        candidate_keywords = []
        for i, sentence in enumerate(sentences):
            curr_keyword = []
            tokens = wordpunct_tokenize(sentence)
             # normalize case and remove punctuation
            words = [w.lower() for w in tokens if w.isalnum()]
            for word in words:
                if word not in self.stopwords:
                    curr_keyword.append(word)
                else:
                    if curr_keyword != []:
                        candidate_keywords.append({
                            'keyword_list': curr_keyword,
                            'sentence_num': i,
                        })
                        curr_keyword = []
        return candidate_keywords

    def generate_keyword_rank(self, keywords):
        '''
        gets the average rank of a keyword, which is degree/frequency
        '''
        keywords_degree, freq = self._get_degrees_of_keywords(keywords)
        ranked_keywords = []
        for k in keywords:
            keyword_score = 0.0
            for word in k['keyword_list']:
                keyword_score += keywords_degree[word] / float(freq[word])
            ranked_keywords.append({
                'score': keyword_score,
                'keyword_phrase': ' '.join(k['keyword_list']),
                'sentence_num': k['sentence_num']
            })

        ordered_keywords = sorted(ranked_keywords, key=lambda x: x['score'], reverse=True) # sorts keywords by rank
        return ordered_keywords

    def _get_degrees_of_keywords(self, keywords):
        '''
        finds the degree and frequency of every individual word
        in the keywords by building a co-occurrance graph
        degree is defined by the number of words in a keyword that includes the word itself
        frequency is how often the word appears
        '''
        degrees = {}
        frequency = {}
        for k in keywords:
            for word in k['keyword_list']:
                if word in degrees:
                    degrees[word] += len(k['keyword_list'])
                    frequency[word] += 1
                else:
                    degrees[word] = len(k['keyword_list'])
                    frequency[word] = 1

        return degrees, frequency
