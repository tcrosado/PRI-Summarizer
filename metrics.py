from nltk.tokenize import sent_tokenize
from nltk.metrics.scores import (precision, recall)
import numpy
import re

def getFilename(fullPath):
    fileName = re.findall(r"/Text/(.*?)\.txt",fullPath)
    if(fileName != []):
        return fileName[0]
    else:
        return ""

def calculateF1score(precision,recall):
    return 2 * (precision * recall) / (precision + recall)

def calculateAveragePrecision(index,precisionList,relevanceList):
    ''' relevanceList is a binary list
    '''
    return sum(precisionList[i] for i in range(index) if relevanceList[i] == 1)/sum(relevanceList)


def calculateMAP(averagePrecisionList):
    return numpy.mean(averagePrecisionList)


def getMetrics(fileName,resultPath,expectedPath,enc):
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
