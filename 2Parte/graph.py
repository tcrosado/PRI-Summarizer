class Graph(object):

	def __init__(self):
		self.graph = dict()

	def addEdge(self,node1,node2):
		if(node1 in self.graph.keys() and self.graph[node1] != []):
			self.graph[node1].append(node2)
		else:
			self.graph[node1] = [node2]	

		if(node2 in self.graph.keys() and self.graph[node2] != []):
			self.graph[node2].append(node1)
		else:
			self.graph[node2] = [node1]	

	def __str__(self):
		return str(self.graph)