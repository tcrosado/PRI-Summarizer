from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from  nltk import pos_tag
from  nltk import RegexpParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_files
from graph import Graph
from metrics import getMetrics
from metrics import getFilename

enc_PT = 'ISO-8859-15'
enc_US = 'UTF-8'
enc = enc_PT

pathFiles = "../../TeMario/"
pathExpectedFiles = "../../idealTeMario/"

files = load_files(pathFiles)

summaries = []

def getOutputFilePath(fileName):
    return pathFiles+"Text/"+fileName + '.out'

def getExpectedFilePath(fileName):
    return pathExpectedFiles+"Ext-"+fileName+".txt"



def getSentencesFromFile(filePath):
    file = open(filePath, 'r', encoding=enc)

    document = file.read()
    # Turn into sentences
    return sent_tokenize(document)

def calcPageRankWPrior(nodeName,pageRank,graph,damping,priorFunc,weightFunc):

    referingSentences = graph.getReferingLinks(nodeName)

    if referingSentences != []:
        priorSum = 0
        for referingSent in referingSentences:
            priorSum += priorFunc(referingSent)
        if priorSum == 0:
            pr = 0
        else:
            pr = damping * (priorFunc(nodeName)/priorSum)
    else:
        pr = 0


    relatedSentences = graph.getReferedLinks(nodeName)
    weights = 0
    for relatedSent in relatedSentences:
        weightPageRanks = pageRank[relatedSent]*weightFunc(nodeName,relatedSent)

        weightSumLinks  = 0
        relatedSentLinks = graph.getReferedLinks(relatedSent)
        for relatedSecSent in relatedSentLinks:
            weightSumLinks += weightFunc(relatedSent,relatedSecSent)

        weights+= weightPageRanks / weightSumLinks

    pr += (1 - damping)*weights

    return pr

def getNounPhrases(sentences):
    # Should remove stopwords
    nounPhrases = []
    words = [word_tokenize(sentence) for sentence in sentences]
    taggedWords = [pos_tag(word) for word in words]
    for sent in taggedWords:
        grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}"""
        parser = RegexpParser(grammar)
        sentence = parser.parse(sent)

        grammar = "NP: {<NN>+}"
        parser = RegexpParser(grammar)
        classTree = parser.parse(sentence)

        flatTree = []
        for leaf in classTree:
            flatTree.append(leaf)

        for leaf in flatTree:
            if isinstance(leaf,Tree) and leaf.label() == 'NP':
                nounPhrase= ''
                for (word,cls) in leaf.leaves():
                    nounPhrase+=" "+word
                nounPhrases.append(nounPhrase)

    return nounPhrases

def countFrequence(sentences):
    freq = dict()
    for sentence in sentences:
        if sentence in freq:
            freq[sentence]+=1
        else:
            freq[sentence]=1
    return freq
        


for filePath in files.filenames:
    fileName = getFilename(filePath)
    #Creating a sentence list
    sentList = getSentencesFromFile(filePath)

    #Getting noun list for prior prob calculator
    words = [word_tokenize(sentenceDoc) for sentenceDoc in sentList]
    taggedWords = [pos_tag(word) for word in words]
    nouns = []
    for sent in taggedWords:
        for word,pos in sent:
            if(pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.append(word)

    #Nouns with more relevance at last so have more weight (index)
    nouns = tuple(sorted(countFrequence(nouns)))

    
    # ################# Weight assignment ################
    def getConsineSimFromSent(sentence1,sentence2):
            vectorSpace = TfidfVectorizer(encoding=enc)
            resultSentences = vectorSpace.fit_transform(sentList)
            i = sentList.index(sentence1)
            j = sentList.index(sentence2)
            return cosine_similarity(resultSentences[i],resultSentences[j])[0][0]

    def getNounPhraseBasedWeight(sentence1,sentence2):
        # weight = number of noun phrases shared between the sentences

        npSentence1 = getNounPhrases([sentence1])
        npSentence2 = getNounPhrases([sentence2])

        weight = 1
        for np in npSentence1:
            if np in npSentence2:
                weight+=1

        return weight



    # ################ Prior Probability assignment ###############
    def getCosineSimSentenceDocument(sentence):
        ''' sentence -> str '''
        vectorSpace = TfidfVectorizer(encoding=enc)
        f = open(filePath, 'r', encoding=enc)
        document = f.read()
        resultDoc = vectorSpace.fit_transform([document])
        resultSentence = vectorSpace.transform(sentList)
        i = sentList.index(sentence)
        return cosine_similarity(resultSentence[i],resultDoc)

    def getNounBasedProbability(sentence):
        ''' sentence -> str '''
        
        wordsSentence = word_tokenize(sentence) 

        weight = 1
        for word in wordsSentence:
            if word in nouns:
                weight += nouns.index(word)+1
        return weight


    #################################################################

    def createGraph(sentenceList,treshold):
        graph = Graph()

        for i in range(len(sentenceList)):
            for j in range(i+1,len(sentenceList)):
                vectorSpace = TfidfVectorizer(encoding=enc)
                resultSentences = vectorSpace.fit_transform(sentenceList)

                sim = cosine_similarity(resultSentences[i],resultSentences[j])

                if(sim>=treshold):
                    graph.addBiEdge(sentList[i],sentList[j])
        return graph


    treshold = 0.2
    graph = createGraph(sentList,treshold)

    pageRank = dict()

    for sentence in sentList:
        pageRank[sentence] = 1/len(sentList)

    for sentence in sentList:
        #pr =calcPageRankWPrior(sentence,pageRank,graph,0.15,
        #    sentList.index,getConsineSimFromSent)

        #pr =calcPageRankWPrior(sentence,pageRank,graph,0.15,
        #    sentList.index,getNounPhraseBasedWeight)

        #pr =calcPageRankWPrior(sentence,pageRank,graph,0.15,
        #   getCosineSimSentenceDocument,getConsineSimFromSent)
        #pr =calcPageRankWPrior(sentence,pageRank,graph,0.15,
        #  getCosineSimSentenceDocument,getNounPhraseBasedWeight)
        pr =calcPageRankWPrior(sentence,pageRank,graph,0.15,
           getNounBasedProbability,getNounPhraseBasedWeight)

        pageRank[sentence] = pr

    #FIXME: Its joining everything into summaries
    summaries += sorted(pageRank, key=pageRank.__getitem__)[:5]
    print(summaries)
    #resultFilePath = getOutputFilePath(fileName+"D")
    #expectedFilePath = getExpectedFilePath(fileName)
    #with open(resultFilePath,"w", encoding = enc) as f:
    #    f.write("\n".join(summaries))

    #getMetrics(fileName,resultFilePath,expectedFilePath,enc)




