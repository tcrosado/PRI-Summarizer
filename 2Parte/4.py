from urllib.request import urlopen
import time
import feedparser			#lib alternativa


feeds=['http://rss.nytimes.com/services/xml/rss/nyt/World.xml','http://rss.cnn.com/rss/edition_world.rss','http://feeds.washingtonpost.com/rss/world','http://www.latimes.com/world/rss2.0.xml']

#############################################lib alternative and better than the one from the teacher####################################################################
final=[]			##this has a format [[title,summary][tittle,summary]]
def tryer ():
	for i in feeds:
		feed=feedparser.parse(i)
		for o in feed['entries']:
			aux_list=[]
			aux_list.append(o['title'])
			aux_list.append(o['summary'])######we stil need to strip some html stuff
			final.append(aux_list)
		print(final)
		time.sleep(1) 


tryer()
#####################################################################################################################