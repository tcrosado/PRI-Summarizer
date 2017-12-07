from sklearn import datasets
from sklearn.linear_model import Perceptron

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.datasets import load_files
import re
import codecs
import operator

#import networkx as nx
#from networkx.algorithms import closeness_centrality

import numpy

pathFiles = "TeMario/"

pathExpectedFiles = "idealTeMario/"

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT

files = load_files(pathFiles,encoding = enc, decode_error='ignore')

summary = load_files(pathExpectedFiles,encoding = enc, decode_error='ignore')

vectorSpace = TfidfVectorizer(encoding=enc)

#vectorSpace = TfidfVectorizer(stop_words="english")

scoreSentences = dict()



#-----------------------------------------------------------------------------------------------------#
def getFilename(fullPath):
    fileName = re.findall(r"/(.*)/(.*?)\.txt",fullPath)
    if(fileName != []):
        return fileName[0]
    else:
        return ""

def join_doc(document):
	sentences = sent_tokenize(document)

	return sentences


def tf_idf_doc(document,resultDocument):	
	# Turn into sentences
	sentences = sent_tokenize(document)

	# Parse Title as sentence
	firstSentence = sentences[0]
	sentences.remove(firstSentence)

	sentences = firstSentence.split("\n") + sentences

	#TFIDF in each sentence
	resultSentence = vectorSpace.transform(sentences)

	lineScores = dict()
	allSentenceScores = []
	for i in range(len(sentences)):
		similarity = cosine_similarity(resultSentence[i:i+1],resultDocument)[0]
		lineScores[i] = similarity
	
	return lineScores


def positionDocument(document):
	sentences = sent_tokenize(document)	

	firstSentence = sentences[0]
	sentences.remove(firstSentence)

	sentences = firstSentence.split("\n") + sentences

	pos = dict()
	for i in range(len(sentences)):
		pos[i] = i+1

	return pos


def is_in_summary(document,summary):
	sentences = sent_tokenize(document)	

	#sentences = [sentence.translate(None,string.punctuation) for sentence in sentences]

	#print (sentences)

	final_summary_sentences = sent_tokenize(summary)

	#final_summary_sentences =[sentence.translate(None,string.punctuation) for sentence in final_summary_sentences]
	
	#print (final_summary_sentences)

	final_summary = dict()
	for i in range(len(sentences)):
		if sentences[i] in final_summary_sentences:
			final_summary[i] = 1
		else:
			final_summary[i] = 0

	return final_summary

#---------------------------------------------START OF TRAINING----------------------------------------------------------------#

def perceptron_train(files, summary):
	scoreFileSentences = dict()
	position = dict()
	for file in files.filenames:
		fileName = getFilename(file)
		#print (fileName)

		f = codecs.open(file, 'r', encoding = enc, errors = 'ignore')
		document = f.read()


		resultDocument = vectorSpace.fit_transform([document])
		
		scoreFileSentences[fileName] = tf_idf_doc(document, resultDocument)
		position[fileName] = positionDocument(document)

		#print (position)

	in_summary = dict()



	for file in summary.filenames:
		fileName = getFilename(file)
		#print (fileName)

		f = codecs.open(file, 'r', encoding = enc, errors = 'ignore')
		summary = f.read()
		f.close()
		
		'''
		print (pathFiles)
		print(fileName[0])
		print(fileName[1][4:])'''

		f = codecs.open(pathFiles+fileName[0]+'/'+fileName[1][4:]+'.txt', 'r', encoding = enc, errors = 'ignore')
		document = f.read()
		f.close()

		in_summary[(fileName[0],fileName[1][4:])] = is_in_summary(document, summary)

		#print (in_summary)

	matrix_feature = []
	for doc in scoreFileSentences:
		for i in range(len(doc)):
			matrix_feature.append([scoreFileSentences[doc][i], position[doc][i]])

	array_y = []
	for doc in in_summary:
		for i in range(len(doc)):
			array_y.append(in_summary[doc][i])

	training_array = numpy.array(matrix_feature)
	training_y = numpy.array(array_y)

	ppt = Perceptron(random_state = 0)
	ppt.fit(training_array,training_y)

	return ppt

#----------------------------------------------END OF TRAINING--------------------------------------------------#	


def run(files, summary):
	ppt = perceptron_train(files, summary)

	all_sentences = dict()
	scoreFileSentences = dict()
	position = dict()
	for file in files.filenames:
		fileName = getFilename(file)
		#print (fileName)

		f = codecs.open(file, 'r', encoding = enc, errors = 'ignore')
		document = f.read()

		all_sentences[fileName] = join_doc(document)

		resultDocument = vectorSpace.fit_transform([document])
		
		scoreFileSentences[fileName] = tf_idf_doc(document, resultDocument)
		position[fileName] = positionDocument(document)

		#print (position)

	stcs = []
	for item in all_sentences:
		for i in range(len(all_sentences[item])):
			#print (all_sentences[item])
			stcs.append(all_sentences[item][i])

	#print (len(stcs))

	matrix = []
	for doc in scoreFileSentences:
		for i in range(len(all_sentences[doc])):
			matrix.append([scoreFileSentences[doc][i], position[doc][i]])

	#print (matrix)

	testing_matrix = numpy.array(matrix)
	#print (testing_matrix)
	results = ppt.decision_function(testing_matrix)
	#print (results.shape)


	result_dict = {}
	for i in range(0,len(results)):
			result_dict[stcs[i]] = results[i]

	sorted_results = sorted(result_dict.items(), key=operator.itemgetter(1))

	for item in sorted_results[-5:]:
		print (item[0])

run(files, summary)