from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
file = open("example.in")

document = file.read()
# Turn into sentences
sentences = sent_tokenize(document)

#TFIDF in each sentence
vectorSpace = TfidfVectorizer(stop_words="english")
resultSentence = vectorSpace.fit_transform(sentences)

#print(resultSentence)

#TFIDF of document
resultDocument = vectorSpace.fit_transform([document])

#print (resultDocument)

scoreSentences = dict()
for i in range(0,len(sentences)):
    scoreSentences[i]=float(cosine_similarity(resultSentence[i:i+1],resultDocument))

sortedScore = sorted(scoreSentences.items(),key=operator.itemgetter(1),reverse=True)

keySentenceIDF = sortedScore[0:3]
keySentences = sorted(keySentenceIDF)

for key in keySentences:
    print(sentences[key[0]])
