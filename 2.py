from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_files
import operator
import re
import os
import numpy
from Ex1Lib import getSenteceBasedSummary
from metrics import *

pathFiles = "../TeMario/"
pathExpectedFiles = "../idealTeMario/"

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT



def getOutputFilePath(fileName):
    return pathFiles+"Text/"+fileName + '.out'

def getExpectedFilePath(fileName):
    return pathExpectedFiles+"Ext-"+fileName+".txt"


def getNthMaxScores(n,scores):
	result = []
	scoreCopy = scores.copy()
	for i in range(n):
		maxNumber = max(scoreCopy)
		result.append(maxNumber)
		scoreCopy.remove(maxNumber)
	return result


def selectNthBestLines(n,fileName,allFileScores,scoreFileSentences):
	fileMaxScores = getNthMaxScores(5,allFileScores[fileName])

	lineScores = scoreFileSentences[fileName]
	selectedLines = []
	for lineNumber in lineScores.keys():
		lineMaxScore = max(lineScores[lineNumber])
		if lineMaxScore  in fileMaxScores:
			fileMaxScores.remove(lineMaxScore)
			if lineNumber not in selectedLines:
				selectedLines.append(lineNumber)

	return sorted(selectedLines)


def getDocumentBasedSummary(nrSummarySentences,vectorSpace,files):

    #TFIDF in collection
    resultCollection = vectorSpace.fit_transform(files.data)

    sentences = dict()
    scoreFileSentences = dict() # cant put in function
    allFileScores = dict()
    for file in files.filenames:
        fileName = getFilename(file)
        f = open(file, 'r', encoding=enc)
        document = f.read()
        # Turn into sentences
        sentences[fileName] = sent_tokenize(document)

        # Parse Title as sentence
        firstSentence = sentences[fileName][0]
        sentences[fileName].remove(firstSentence)

        sentences[fileName] = firstSentence.split("\n") + sentences[fileName]

        #TFIDF in each sentence
        resultSentence = vectorSpace.transform(sentences[fileName])

        lineScores = dict()
        allSentenceScores = []
        for i in range(len(sentences[fileName])):
            similarity = cosine_similarity(resultSentence[i:i+1],resultCollection)[0]
            allSentenceScores += list(similarity)
            lineScores[i] = similarity
        scoreFileSentences[fileName] = lineScores
        allFileScores[fileName] = allSentenceScores


    summaries = dict()
    for fileName in scoreFileSentences.keys():

        selectedLines = selectNthBestLines(nrSummarySentences,fileName,allFileScores,scoreFileSentences)
        filePath = getOutputFilePath(fileName)

        summary = []
        for lineNumber in selectedLines:
            summary.append(sentences[fileName][lineNumber]+"\n")

        summaries[fileName]=tuple(summary)

    return summaries


files = load_files(pathFiles)

vectorSpace = TfidfVectorizer(encoding=enc)
summaryDocBased = getDocumentBasedSummary(5,vectorSpace,files)


for filePath in files.filenames:

    # Exporting Document based results to file
    fileName = getFilename(filePath)

    outputDocPath = getOutputFilePath(fileName+"D")
    with open(outputDocPath, 'w', encoding=enc) as file:
        file.write("".join(summaryDocBased[fileName]))

    #Sentence based approach
    vectorSpace = TfidfVectorizer(encoding=enc)
    summarySentenceBased = getSenteceBasedSummary(3,vectorSpace,filePath,enc)

    # Exporting Sentence based results to file
    outputSenPath = getOutputFilePath(fileName+"S")
    with open(outputSenPath, 'w', encoding=enc) as file:
        file.write("\n".join(summarySentenceBased))




# Precision, recall, F1 score and MAP calculations

precisionDocBased = []
relevanceDocBased = []

precisionSentBased = []
relevanceSentBased = []


for filePath in files.filenames:
    fileName = getFilename(filePath)
    expectedFilePath = getExpectedFilePath(fileName)

    #Metrics for Document Based approach
    print("Document Based")
    resultFilePath = getOutputFilePath(fileName+"D")
    resultDocBased = getMetrics(fileName,resultFilePath,expectedFilePath,enc)


    precisionDocBased += [resultDocBased["precision"]]
    relevanceDocBased += [resultDocBased["relevance"]]

    #Metrics for Sentece Based approach
    print("Sentece Based")
    resultFilePath = getOutputFilePath(fileName+"S")
    resultSentBased = getMetrics(fileName,resultFilePath,expectedFilePath,enc)

    precisionSentBased += [resultSentBased["precision"]]
    relevanceSentBased += [resultSentBased["relevance"]]

avgPrecisionDocBased = calculateAveragePrecision(len(precisionDocBased),precisionDocBased,relevanceDocBased)
avgPrecisionSentBased = calculateAveragePrecision(len(precisionSentBased),precisionSentBased,relevanceSentBased)
print("Mean Average Precision: ",calculateMAP([avgPrecisionDocBased,avgPrecisionSentBased]))

#Clean Up
os.system("rm "+pathFiles+"Text/*.out")