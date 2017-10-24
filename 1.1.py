from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import nltk
nltk.download('punkt')
from collections import defaultdict, Counter


with open('data.txt', 'r') as myfile:			##Read a file
    data=myfile.read().lower()


data_words=Counter(nltk.word_tokenize(data)) ##Dictionary of the number that the words repeat themselfs in the Doc
data_token=nltk.sent_tokenize(data)	##Split the sentences

vec = DictVectorizer()
vec.fit_transform(data_words).toarray()

#vectorizer = TfidfVectorizer( use_idf=True )
#trainvec = vectorizer.fit_transform(myfile)
#testvec = vectorizer.transform(test.data)

print vec.get_feature_names()

#not sure what I'm doing here