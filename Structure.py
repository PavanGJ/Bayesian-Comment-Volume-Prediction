################################################################################
#
#	Date 			Name		Description
#
#	23-Mar-2017     Pavan Joshi	Created Data Structure to handle the Graph
#					structure to be fed to libpgm
#
#
################################################################################

class Structure:
	structure = dict()

	def __init__(self):
		self.structure['V'] = list()
		self.structure['E'] = list()
		self.structure['V_type'] = list()

	def add_edge(self,edge,types=["d","d"]):
		if edge not in self.structure['E']:
			idx = 0
			for vertex in edge:
				if vertex not in self.structure['V']:
					self.structure['V'].append(vertex)
					self.structure['V_type'].append(types[idx])
					print('Vertex %s added to the network'%vertex)
				idx = idx + 1
			self.structure['E'].append(edge)
			print('Edge added to the network')
		else:
			raise ValueError('Edge already present in the network')

	def add_vertex(self,vertex,type="d"):
		if vertex not in self.structure['V']:
			self.sturcture['V'].append(vertex)
			self.structure['V_type'].append(type)
			print('Vertex %s added to the network'%vertex)
		else:
			raise ValueError('Vertex already present in the network')

	def get_children(self,vertex):
		if vertex not in self.structure['V']:
			raise ValueError('Vertex %s not in the network'%vertex)
		children = [edge[1] for edge in self.structure['E'] if edge[0]==vertex]
		return children

	def get_parents(self,vertex):
		if vertex not in self.structure['V']:
			raise ValueError('Vertex %s not in the network'%vertex)
		parents = [edge[0] for edge in self.structure['E'] if edge[1]==vertex]
		return parents

	def get_structure(self):
		return self.structure

	def get_vertex_type(self,vertex):
		if vertex not in self.structure['V']:
			raise ValueError('Vertex %s not in the network'%vertex)
		for idx in range(len(self.structure['V'])):
			if self.structure['V'][idx] == vertex:
				return self.structure['V_type'][idx]
