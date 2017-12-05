from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_files
from graph import Graph

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT

pathFiles = "../../TeMario/"


def getSentencesFromFile(filePath):
    file = open(filePath, 'r', encoding=enc)

    document = file.read()
    # Turn into sentences
    return sent_tokenize(document)

def calcPageRankWPrior(nodeName,pageRank,graph,damping,priorFunc,weightFunc):

    referingSentences = graph.getReferingLinks(nodeName)

    if referingSentences != []:
        priorSum = 0
        for referingSent in referingSentences:
            priorSum += priorFunc(referingSent)
        if priorSum == 0:
            pr = 0
        else:
            pr = damping * (priorFunc(nodeName)/priorSum)
    else:
        pr = 0


    relatedSentences = graph.getReferedLinks(nodeName)
    weights = 0
    for relatedSent in relatedSentences:
        weightPageRanks = pageRank[relatedSent]*weightFunc(nodeName,relatedSent)

        weightSumLinks  = 0
        relatedSentLinks = graph.getReferedLinks(relatedSent)
        for relatedSecSent in relatedSentLinks:
            weightSumLinks += weightFunc(relatedSent,relatedSecSent)

        weights+= weightPageRanks / weightSumLinks

    pr += (1 - damping)*weights

    return pr




files = load_files(pathFiles)

fileName = files.filenames[0]

#Creating a sentence list
sentList = getSentencesFromFile(fileName)

def getConsineSimFromSent(sentence1,sentence2):
        vectorSpace = TfidfVectorizer(encoding=enc)
        resultSentences = vectorSpace.fit_transform(sentList)
        i = sentList.index(sentence1)
        j = sentList.index(sentence2)
        return cosine_similarity(resultSentences[i],resultSentences[j])


#Creating graph
treshold = 0.2
graph = Graph()

for i in range(len(sentList)):
    for j in range(i+1,len(sentList)):
        vectorSpace = TfidfVectorizer(encoding=enc)
        resultSentences = vectorSpace.fit_transform(sentList)

        sim = cosine_similarity(resultSentences[i],resultSentences[j])

        if(sim>=treshold):
            graph.addBiEdge(sentList[i],sentList[j])

pageRank = dict()

for sentence in sentList:
    pageRank[sentence] = 1/len(sentList)

for sentence in sentList:
    pr =calcPageRankWPrior(sentence,pageRank,graph,0.15,
        sentList.index,getConsineSimFromSent)

    pageRank[sentence] = pr if pr==0 else pr[0][0]


print(sorted(pageRank, key=pageRank.__getitem__)[:5])





