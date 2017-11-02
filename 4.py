from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import glob

resume=dict()	

number=0

for p in glob.glob('../TeMario/*.txt'):

	file = open(p)

	document = file.read()
	# Turn into sentences
	sentences = sent_tokenize(document)

	#TFIDF in each sentence
	vectorSpace = TfidfVectorizer(stop_words="english")
	resultSentence = vectorSpace.fit_transform(sentences)

	#print(resultSentence)

	#TFIDF of document
	resultDocument = vectorSpace.fit_transform([document])

	#print (resultDocument)

	scoreSentences = dict()
	for i in range(0,len(sentences)):
	    scoreSentences[i]=float(cosine_similarity(resultSentence[i:i+1],resultDocument))

	sortedScore = sorted(scoreSentences.items(),key=operator.itemgetter(1),reverse=True)

	keySentenceIDF = sortedScore[0:3]
	keySentences = sorted(keySentenceIDF)

	total=[]
	total.append(sortedScore[0])
	for a in sortedScore:

		if len(total)<5 and total[-1][1]!=a[1]:
			total.append(a)
		
		if total ==5:
			break
	
	resume[number]=total
	number+=1
print(resume)