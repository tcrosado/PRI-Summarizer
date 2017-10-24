import nltk
nltk.download('punkt')
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open('data.txt', 'r') as myfile:			##Read a file
    data=myfile.read().lower()


#data_words=Counter(nltk.word_tokenize(data)) ##Dictionary of the number that the words repeat themselfs in the Doc
data_words = nltk.word_tokenize(data)
data_token=nltk.sent_tokenize(data)	##Split the sentences

final_list=[]
word_list=Counter(nltk.word_tokenize(data))
for key, value in word_list.iteritems():
	final_list.append(key)
vec = CountVectorizer(vocabulary=final_list)
sentences = vec.fit_transform(data_token).toarray()
print sentences										####### creates a matrix where it put the number of times the words repeat of a a vocabulary wich in our case is the full doc

transformer = TfidfTransformer(smooth_idf=True)
tfidf = transformer.fit_transform(sentences).toarray()

print tfidf
###############################################################################################################
full_document_matrix=vec.fit_transform(data_words)   ## will be needed to the cosine of similarity
print full_document_matrix
tfidf_full_document=transformer.fit_transform(full_document_matrix).toarray()
###########################################################################################################
######idf wow


print cosine_similarity(tfidf[0:],tfidf_full_document)




### ha calculos aqui a ser feitos mal