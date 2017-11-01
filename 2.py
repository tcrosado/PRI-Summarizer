from nltk.tokenize import sent_tokenize
from nltk.metrics.scores import (precision, recall)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_files
import operator
import re
import os

def getFilename(fullPath):
	fileName = re.findall(r"/Text/(.*?)\.txt",fullPath)
	if(fileName != []):
		return fileName[0]
	else:
		return ""

def getNthMaxScores(n,scores):
	result = []
	scoreCopy = scores.copy()
	for i in range(n):
		maxNumber = max(scoreCopy)
		result.append(maxNumber)
		scoreCopy.remove(maxNumber)
	return result

def calculateF1score(precision,recall):
	return 2 * (precision * recall) / (precision + recall)



pathFiles = "../TeMario/"
pathExpectedFiles = "../idealTeMario/"
files = load_files(pathFiles)

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT


#TFIDF in collection
vectorSpace = TfidfVectorizer(encoding=enc)
resultCollection = vectorSpace.fit_transform(files.data)

sentences = dict()
scoreFileSentences = dict()
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




for fileName in scoreFileSentences.keys():
	maxScores = getNthMaxScores(5,allFileScores[fileName])
	
	lineScores = scoreFileSentences[fileName]
	selectedLines = [] 
	for lineNumber in lineScores.keys():
		lineMaxScore = max(lineScores[lineNumber])
		if lineMaxScore  in maxScores:
			maxScores.remove(lineMaxScore)
			if lineNumber not in selectedLines:
				selectedLines.append(lineNumber)

	sortedLines = sorted(selectedLines)
	filePath = pathFiles+"Text/"+fileName + '.out'
	output = open(filePath, 'w', encoding=enc)

	for lineNumber in sortedLines:
		#needs new line so nltk can split sentences correctly
		output.write(sentences[fileName][lineNumber]+"\n")
	
	output.close()

# Here should be the precision, recall and F1 and MA Pcalculations

for fileName in scoreFileSentences.keys():
	output = open(pathFiles+"Text/"+fileName+".out",'r', encoding=enc)
	outputResult = output.read()
	outputSentences = set(sent_tokenize(outputResult))
	output.close()

	expected = open(pathExpectedFiles+"Ext-"+fileName+".txt",'r', encoding=enc)
	expectedResult = expected.readlines()
	expectedSentences = []
	for line in expectedResult:
		expectedSentences += [line[:-1]]

	expectedSentences = set(expectedSentences)
	expected.close()
	
	recallResult = recall(expectedSentences,outputSentences)
	precisionResult = precision(expectedSentences,outputSentences)
	print("#")
	print("file: ",fileName)
	print("recall: ",recallResult)
	print("precision: ",precisionResult)
	if recallResult != 0 and precisionResult != 0:
		print("F1 Score: ",calculateF1score(precisionResult,recallResult))

#Clean Up
os.system("rm "+pathFiles+"Text/*.out")