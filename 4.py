from urllib.request import urlopen
import time



def openPage(url):
	site = urlopen(url)
	content = site.read()

	site.close()
	return content



def getRssFeeds():
	feeds=['http://www.nytimes.com/services/xml/rss/index.html','http://edition.cnn.com/services/rss/','https://www.washingtonpost.com/rss-feeds/2014/08/04/ab6f109a-1bf7-11e4-ae54-0cfe1f974f8a_story.html','http://www.latimes.com/local/la-los-angeles-times-rss-feeds-20140507-htmlstory.html']
	for i in feeds:
		feed=openPage(i)
		
		time.sleep(1) 				#waits 1 second betweeen each request its nedded
		print(feed)

getRssFeeds()