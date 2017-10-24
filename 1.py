## ALL wrong maybe we can use some of these code later on####


import nltk
nltk.download('punkt')
from collections import defaultdict, Counter
import math
#import sklearn

with open('data.txt', 'r') as myfile:			##Read a file
    data=myfile.read().lower()
data_words=Counter(nltk.word_tokenize(data)) ##Dictionary of the number that the words repeat themselfs in the Doc
data_token=nltk.sent_tokenize(data)	##Split the sentences

DicDocument=defaultdict(list)
DicSentence=defaultdict(list)

###################################################################################################
count=0	
###################################################################################################	
				##get the frequency of each word in its sentence
					#Number the sentences

for a in data_token:
	dic_toadd=defaultdict(list)
	aux_list=Counter(nltk.word_tokenize(a))

	for key,value in aux_list.iteritems():
		dic_toadd[key].append(math.log(len(data)/aux_list.get(key)))
	DicSentence[count].append(dic_toadd)
	count=count+1

for key, value in data_words.iteritems(): 			#		## Iterate and put it in a list	
	DicDocument[key].append(math.log(len(data)/value))


#print data_words
print DicSentence
print DicDocument

###############################################################################################3
#sklearn.metrics.pairwise.cosine_similarity