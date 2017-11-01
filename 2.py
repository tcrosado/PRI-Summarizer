from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_files
import operator
import re


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
	
	filePath = pathFiles+"/Text/"+fileName + '.out'
	output = open(filePath, 'w', encoding=enc)

	for lineNumber in sortedLines:
		output.write(sentences[fileName][lineNumber])
	
	output.close()

# Here should be the precision, recall and F1 and MAP calculations