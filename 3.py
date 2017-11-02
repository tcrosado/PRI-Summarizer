from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import operator
import math

file = open("example.in")

document = file.read()

sentences = sent_tokenize(document)

counter = CountVectorizer(stop_words='english') 
counter.fit_transform([document])
number_of_training_words = len(counter.vocabulary_)


vectorSpace = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
resultSentence = vectorSpace.fit_transform(sentences)

def calculate_bm25(matrix,total_sentence):
    indices = matrix.indices
    matrix_array = matrix.toarray()
    bm25 = []
    i = 0
    avgdl = number_of_training_words / total_sentence
    k1 = 1.2
    b = 0.75
    len(indices)
    for index in indices:
        frequencies = matrix.data
        current_idf = calculate_idf(index,matrix_array,total_sentence)
        term_freq = frequencies[i]
        bm25_value = (current_idf *((term_freq * (k1 + 1))  /  (term_freq + k1 * (1 - b + b*(matrix.getnnz()/number_of_training_words)))))
        bm25.append(bm25_value)
        i+=1
    return bm25

def calculate_idf(index, matrix, total_sentence):
    sentences_with_term = sum(row[index] for row in matrix)

    return math.log((total_sentence - sentences_with_term + 0.5)/(sentences_with_term + 0.5))


i = 0
features = vectorSpace.get_feature_names()
indices = resultSentence.indices

bm25 = dict()
for i in range(len(sentences)):
    bm25[i] = calculate_bm25(resultSentence[i:i+1],len(sentences))

sorted_bm25 = sorted(bm25.items(), key=operator.itemgetter(1))

keySentenceIDF = sorted_bm25[0:3]
keySentences = sorted(keySentenceIDF)

for key in keySentences:
    print(sentences[key[0]])
