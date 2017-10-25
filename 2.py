from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_files
import operator
files = load_files("../TeMario/")

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT


#TFIDF in collection
vectorSpace = TfidfVectorizer(encoding=enc)
resultCollection = vectorSpace.fit_transform(files.data)

#print (len(files.filenames))
sentences = []

for file in files.filenames:
	f = open(file, 'r', encoding=enc)
	document = f.read()
	# Turn into sentences
	sentencesDocument = sent_tokenize(document)
	sentences += sentencesDocument
	#TFIDF in each sentence
	resultSentence = vectorSpace.transform(sentencesDocument)

	#print (resultDocument)

	scoreSentences = []
	for i in range(0,len(sentences)):
#######################################################################################
#######################################################################################
	    scoreSentences.append(cosine_similarity(resultSentence[i:i+1],resultCollection)[0])

for i in range(len(files.filenames)):

	scoreDoc = dict(zip(sentences, scoreSentences[i]))

	sortedScore = sorted(scoreDoc.items(),key=operator.itemgetter(1),reverse=True)

#######################################################################################
#######################################################################################

	keySentenceIDF = sortedScore[0:5]
	keySentences = sorted(keySentenceIDF)

	nameFile = files.filenames[i] + '.out'

	output = open(nameFile, 'w', encoding=enc)

	for key in keySentences:
		print(sentences[key[0]])
		outpu.twrite(sentences[key[0]])

	output.close()