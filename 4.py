from urllib.request import urlopen
import time
import feedparser			#lib alternativa

feeds=['http://rss.nytimes.com/services/xml/rss/nyt/World.xml','http://rss.cnn.com/rss/edition_world.rss','http://feeds.washingtonpost.com/rss/world','http://www.latimes.com/world/rss2.0.xml']

def openPage(url):
	site = urlopen(url)
	content = site.read().decode('utf-8')
	site.close()
	return content



def getRssFeeds():
	
	for i in feeds:
		feed=openPage(i)
		time.sleep(1) 				#waits 1 second betweeen each request its nedded
		print(feed)

#getRssFeeds()



#############################################lib alternative and better than the one from the teacher####################################################################
def tryer ():
	for i in feeds:
		feed=feedparser.parse(i)
		print(feed)
		time.sleep(1) 	


tryer()
#####################################################################################################################