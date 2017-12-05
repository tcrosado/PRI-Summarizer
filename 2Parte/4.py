import time
import os
import feedparser			#lib alternativa
import re
from goose3 import Goose
####
from dominate import document
from dominate.tags import *
####

feeds=['http://rss.nytimes.com/services/xml/rss/nyt/World.xml','http://rss.cnn.com/rss/edition_world.rss','http://feeds.washingtonpost.com/rss/world','http://www.latimes.com/world/rss2.0.xml']

#####################################################################################################################
def createfiles(feed,name,content):
	result = re.search('%s(.*)%s' % ('http://', '.com/'), feed).group(1)
	print(name)
	if not os.path.exists('./toParse/'+result):
		os.makedirs('./toParse/'+result)
	f = open('./toParse/'+result+'/'+name,'w')
	f.write(content)
	f.close()
###################################################################
def fullNews(link,feed):

	g = Goose()
	try:
		article = g.extract(url=link)
		createfiles(feed,article.title,article.cleaned_text)
	except:
		print('error')
	
########################## Generate HTML ############################
def generateHtml():

	with document(title='News Resume') as doc:
	    h1('Feed')
	    with div():
        	attr(cls='pretty')
        	p()

	with open('gallery.html', 'w') as f:
	    f.write(doc.render())

#############################################lib alternative and better than the one from the teacher####################################################################

def tryer ():
	for i in feeds:
		feed=feedparser.parse(i)
		for o in feed['entries']:
			fullNews(o['link'],i)
		time.sleep(1) 

tryer()





