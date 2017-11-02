from nltk.tokenize import sent_tokenize
from nltk.metrics.scores import (precision, recall)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_files
import operator
import re
import os

pathFiles = "../TeMario/"
pathExpectedFiles = "../idealTeMario/"

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT



def getFilename(fullPath):
	fileName = re.findall(r"/Text/(.*?)\.txt",fullPath)
	if(fileName != []):
		return fileName[0]
	else:
		return ""

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


def getDocumentBasedSummary(vectorSpace,files,nrSummarySentences):

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





def calculateF1score(precision,recall):
	return 2 * (precision * recall) / (precision + recall)

def calculateAveragePrecision(index,precisionList,relevanceList):
	''' relevanceList is a binary list
	'''
	return sum(precisionList[i] for i in range(index) if relevanceList[i] == 1)/sum(relevanceList)


def calculateMAP(averagePrecisionList):
	return numpy.mean(averagePrecisionList)


def getMetrics(resultPath,expectedPath):
    output = open(resultPath,'r', encoding=enc)
    outputResult = output.read()
    outputSentences = set(sent_tokenize(outputResult))
    output.close()

    expected = open(expectedPath,'r', encoding=enc)
    expectedResult = expected.readlines()


    #removing newlines for better results
    expectedSentences = []
    for line in expectedResult:
        expectedSentences += [line[:-1]]

    expectedSentences = set(expectedSentences)
    expected.close()

    recallResult = recall(expectedSentences,outputSentences)
    precisionResult = precision(expectedSentences,outputSentences)

    resultString = "File: "+fileName+" Recall: "+str(recallResult)+" Precision: "+str(precisionResult)
    if recallResult != 0 and precisionResult != 0:
        f1Score = calculateF1score(precisionResult,recallResult)
        resultString += " F1 Score: "+str(f1Score)

    print(resultString)

    return {"recall" : recallResult,"precision" : precisionResult,"relevance": 1 if precisionResult!=0 else 0}






#cenas

#need file list
files = load_files(pathFiles)
vectorSpace = TfidfVectorizer(encoding=enc)
summary = getDocumentBasedSummary(vectorSpace,files,5)


for filePath in files.filenames:
    fileName = getFilename(filePath)
    outputPath = getOutputFilePath(fileName)
    with open(outputPath, 'w', encoding=enc) as file:
        file.write("".join(summary[fileName]))



# Precision, recall, F1 score and MAP calculations

precisionList = []
relevanceList = []
# Chage ScoreFileSenetences to file.filename or list of filenames

for filePath in files.filenames:
    fileName = getFilename(filePath)
    expectedFilePath = getExpectedFilePath(fileName)
    resultFilePath = getOutputFilePath(fileName)

    result = getMetrics(expectedFilePath, resultFilePath)

    precisionList += [result["precision"]]
    relevanceList += [result["relevance"]]

avgPrecision = calculateAveragePrecision(len(precisionList),precisionList,relevanceList)
print("Average Precision: ",avgPrecision)

#Clean Up
os.system("rm "+pathFiles+"Text/*.out")