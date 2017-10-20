import nltk
nltk.download('punkt')
from collections import defaultdict, Counter
import math

with open('data.txt', 'r') as myfile:			##Read a file
    data=myfile.read()
data_words=Counter(nltk.word_tokenize(data))
data_token=nltk.sent_tokenize(data)	##Split the sentences

Dic=defaultdict(list)

for key, value in data_words.iteritems(): 			#freq.iteritems():		## Iterate and put it in a list
	word_frequency=0
	for o in data_token:
		if key in o:
			word_frequency=word_frequency+1

	Dic[key].append(math.log(len(data_token)/word_frequency))
