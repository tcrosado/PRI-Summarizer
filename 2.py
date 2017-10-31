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





pathFiles = "../TeMario/"
pathExpectedFiles = "../idealTeMario/"
files = load_files(pathFiles)

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT


#TFIDF in collection
vectorSpace = TfidfVectorizer(encoding=enc)
resultCollection = vectorSpace.fit_transform(files.data)
#print(files.data)
sentences = []
scoreSentences = dict()

for file in files.filenames:
    fileName = getFilename(file)

    f = open(file, 'r', encoding=enc)
    document = f.read()
    # Turn into sentences
    sentencesDocument = sent_tokenize(document)

    sentences += sentencesDocument
	#TFIDF in each sentence
    resultSentence = vectorSpace.transform(sentencesDocument)

    scores = []
    for i in range(0,len(sentencesDocument)):
        scoreSentences[(fileName,i)]=cosine_similarity(resultSentence[i:i+1],resultCollection)[0]

print(len(scoreSentences))



for position in scoreSentences.keys():

    nameFile = pathFiles+"/Text/"+fileName + '.out'

    sortedScore = sorted(scoreSentences.items(),key=operator.itemgetter(1),reverse=True)

    print(sortedScore)

'''
    keySentenceIDF = sortedScore[0:5]
    keySentences = sorted(keySentenceIDF)










    scoreDoc = dict(zip(sentences, scoreSentences))
    print(scoreDoc)
    sortedScore = sorted(scoreDoc.items(),key=operator.itemgetter(1),reverse=True)
    sortedScore.add
	#######################################################################################
	#######################################################################################
    keySentenceIDF = sortedScore[0:5]
	# Order sentence by apperence in document
    keySentences = sorted(keySentenceIDF)

    output = open(nameFile, 'w', encoding=enc)

    for key in keySentences:
        output.write(key[0])

    output.close()



for file in files.filenames:
    filename = getFilename(file)
    if filename == "":
        continue

    output = open(pathFiles+"Text/"+filename+".out",'r', encoding=enc)
    outputResult = output.read()
    expected = open(pathExpectedFiles+"Ext-"+filename+".txt",'r', encoding=enc)
    expectedResult = expected.readlines()
    print("R####################")
    print(sent_tokenize(outputResult))
    print("E------------------------------")
    print(expectedResult)
    output.close()
    expected.close()
'''