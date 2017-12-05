from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from graph import Graph

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_US


def calcPageRank(docOut,damping,iterNr):
    pr = dict()
    
    #setting initial pageRank
    docIds = docOut.keys()
    for docId in docIds:
        pr[docId] = 1/len(docIds)



    for iteration in range(iterNr):
        for docId in docIds:
            pr[docId] = (1 - damping)/len(docIds)
            sumPR = 0
            for link in docOut[docId]:
                sumPR += pr[link]/len(docOut[link])

            pr[docId] += damping * sumPR

    return pr


sentList = []
#Creating a sentence list
with open('pri_cfc.txt') as f:
	line = f.readline() #FIXME testing with just 1 doc
	docId = int(line.split()[0])
	sentences = sent_tokenize(line.split(' ', 1)[1].lower()) 
	
	for sentence in sentences:
		sentList.append(sentence)


#Creating graph
treshold = 0.2
graph = Graph()

for i in range(len(sentList)):
	for j in range(i+1,len(sentList)):
		vectorSpace = TfidfVectorizer(encoding=enc)
		resultSentences = vectorSpace.fit_transform(sentList)	
		sim = cosine_similarity(resultSentences[i],resultSentences[j])
		if(sim>=treshold):
			graph.addEdge(sentList[i],sentList[j])
		
result = calcPageRank(graph.graph,0.15,10)

sortedResult = sorted(result, key=result.__getitem__)
print(sortedResult[:5])



