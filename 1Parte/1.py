from sklearn.feature_extraction.text import TfidfVectorizer
from Ex1Lib import getSenteceBasedSummary

summary = getSenteceBasedSummary(3,TfidfVectorizer(stop_words="english"),"example.in")

print(summary)
