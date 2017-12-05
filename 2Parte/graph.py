class Graph(object):

	def __init__(self):
		self.graph = dict()

	def addBiEdge(self,node1,node2):
		self.addUniEdge(node1,node2)
		self.addUniEdge(node2,node1)

	def addUniEdge(self,node1,node2):
		if(node1 in self.graph.keys() and self.graph[node1] != []):
			self.graph[node1].append(node2)
		else:
			self.graph[node1] = [node2]

		if(node2 not in self.graph.keys()):
			self.graph[node2] = []

	def getReferedLinks(self,node):
		if(node in self.graph.keys()):
			return self.graph[node]
		else:
			return []

	def getReferingLinks(self,node):
		result = []
		if(node in self.graph.keys()):
			for nodeRef in self.graph.keys():
				if(node in self.graph[nodeRef]):
					result.append(nodeRef)
		return result

	def __str__(self):
		return str(self.graph)