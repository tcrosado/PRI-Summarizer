from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator

def getSenteceBasedSummary(nrSummarySentences,vectorSpace,filepath,encoding="utf-8"):
    file = open(filepath,encoding=encoding)

    document = file.read()
    file.close()
    # Turn into sentences
    sentences = sent_tokenize(document)

    #TFIDF in each sentence
    #vectorSpace = TfidfVectorizer(stop_words="english")
    resultSentence = vectorSpace.fit_transform(sentences)

    #TFIDF of document
    resultDocument = vectorSpace.fit_transform([document])

    scoreSentences = dict()
    for i in range(0,len(sentences)):
        scoreSentences[i]=float(cosine_similarity(resultSentence[i:i+1],resultDocument))

    sortedScore = sorted(scoreSentences.items(),key=operator.itemgetter(1),reverse=True)

    keySentenceIDF = sortedScore[0:nrSummarySentences]
    keySentences = sorted(keySentenceIDF)

    summary = []
    for key in keySentences:
        summary.append(sentences[key[0]])

    return  tuple(summary)

