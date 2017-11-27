
import os, os.path
from whoosh.index import create_in
from whoosh.fields import *




schema = Schema(id = NUMERIC(stored=True), content=TEXT)
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
ix = create_in("indexdir", schema)
writer = ix.writer()

with open('pri_cfc.txt') as f:
	for line in f:
		writer.add_document(id=line.split()[0], content=line.split(' ', 1)[1])
		writer.commit()
