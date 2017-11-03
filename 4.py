from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_files
import operator
from metrics import *

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT

def calculateMMR(lamb,document,sentence,selectedSentences):

	simSum = 0
	for selectedSentence in selectedSentences:
		simSum+=cosine_similarity(sentence,selectedSentence)

	return (1 - lamb) * cosine_similarity(sentence,document) - lamb * simSum


resume=dict()

number=0

files = load_files('../TeMario/')

for filePath in files.filenames:

	file = open(filePath, 'r', encoding=enc)

	document = file.read()
	# Turn into sentences
	sentences = sent_tokenize(document)

	#TFIDF in each sentence
	vectorSpace = TfidfVectorizer()
	resultSentence = vectorSpace.fit_transform(sentences)


	#TFIDF of document
	resultDocument = vectorSpace.fit_transform([document])


	lamb = 0.26
	summarySize = 5
	selectedSentences = []
	selectedSentencesScores = []
	while len(selectedSentences) != summarySize:

		score = dict()
		for i in range(0,len(sentences)):
			score[i] = calculateMMR(lamb,resultDocument,resultSentence[i],selectedSentencesScores)

		#max score and select line
		lineNumber = max(score.items(), key=operator.itemgetter(1))[0]
		selectedSentencesScores += [resultSentence[lineNumber]]
		selectedSentences+=[sentences[lineNumber]]



	# Exporting MMR based results to file
	outputMmrPath = filePath[:-4]+"MMR.out"
	with open(outputMmrPath, 'w', encoding=enc) as file:
		file.write("\n".join(selectedSentences))

	expectedResultPath = "../idealTeMario/Ext-"+getFilename(filePath)+".txt"
	getMetrics(getFilename(filePath),outputMmrPath,expectedResultPath,enc)


#Clean Up
os.system("rm "+pathFiles+"Text/*.out")