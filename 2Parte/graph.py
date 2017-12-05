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

	def __str__(self):
		return str(self.graph)