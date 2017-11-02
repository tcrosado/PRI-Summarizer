from sklearn.feature_extraction.text import TfidfVectorizer
from Ex1Lib import getSenteceBasedSummary

summary = getSenteceBasedSummary(TfidfVectorizer(stop_words="english"),"example.in",3)

print(summary)
