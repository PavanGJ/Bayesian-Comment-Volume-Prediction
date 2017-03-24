################################################################################
#
#	Date 		Name		Description
#
#	01-Mar-2017     Anurag Dixit	Initial Draft
#	19-Mar-2017		Anurag Dixit	Added API for conditional probabilities
#	20-Mar-2017  	Anurag Dixit	Bug fix for parsing of data correctly
#	20-Mar-2017		Anurag Dixit	Added changes for Bayesian Model incorporation and Data
#	20-Mar-2017		Anurag Dixit	Added file read for query perform and commented the MPLP
#	21-Mar-2017		Pavan Joshi		Depreciated reduceDimensions function to utilize numpy functions
#	21-Mar-2017		Pavan Joshi		Added API to handle nodes in the network.
#	22-Mar-2017	Anurag Dixit		Added Linear Regression Code for adding cpds for continuos variables
#	24-Mar-2017     Anurag Dixit            Changes done for hybrid bayesian network model compatible data generation
#
#
################################################################################
import os
import csv
import numpy as np
import pandas as pd

from Structure import Structure
from Data import Data
from pgmpy.models import BayesianModel
from pgmpy.inference import Mplp
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from sklearn import linear_model

from construct_graph import Ndata


class BNetwork(Ndata):


	def __init__(self, fname):
		
		Ndata.__init__(self, fname)
		self.fname = fname
		self.model = Structure()
		

	
	def sdalinit__(self,nodeName,parents = [],continuous = False):
		"""
		Parameters
		----------
		nodeName: The name of the node in the bayesian network

		parents	: The parents of the node.
				  Defaut value = []. Signifies no parents

		continuous: Signifies whether the node contains continuous values
				  False - Discrete
				  True - Continuous
				  Default value = False(Discrete)

		Return
		------
		None

		Examples
		--------

		>>> x = Node('X')
		>>> y = Node('Y',parents=['X'])
		>>> z = Node('Z',parents=['X','Y'],continuous=True)

		"""
		self.nodeName = nodeName
		self.parents = np.array(parents)
		self.continuous = continuous
		self.cpd = None

	def add_parent(self,parent):
		self.parents = np.append(self.parents,parent)

	def add_cpd(self,node_values,parent_values):
		"""
		Parameters
		----------

		node_values		: Values that the node takes.
						  This is the discrete values or a variance in case
						  of continuous distribution.

		parent_values	: The parent values
						  Takes a list of number of discrete parent values.
						  Or it takes the beta vector which specifies the
						  mean of the node as a function of the parent nodes.
						  (Includes beta_0, the constant term which is specified
						  at the beginning)

		Return
		------
		None

		Examples
		--------

		>>> x = Node('X')
		>>> x.add_cpd([[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3],
		 									[0.3, 0.7, 0.02, 0.2]], [2,2])
		CPD added successfully

		>>> y = Node('Y',continuous=True)
		>>> y.add_cpd(9.6 , [0.2,-2, 3, 7])
		CPD added successfully

		"""
		if self.continuous:
			if(parent_values.shape[0]!=(self.parents.shape[0]+1)):
				print "Error: Dimension Mismatch"
				return
			self.cpd = LinearGaussianCPD(
							self.nodeName,
							parent_values,
							node_values,
							self.parents)
		else:
			variable_card = np.array(node_values).shape[0]
			if(variable_card==1):
				variable_card = np.array(node_values).shape[1]
			if(len(parent_values) != 0 and
					np.prod(parent_values)!=np.array(node_values).shape[1]):
				print "Error: Dimension Mismatch"
				return
			self.cpd = TabularCPD(
							self.nodeName,
							variable_card,
							node_values,
							self.parents,
							parent_values
							)

		print('CPD added successfully')

	def get_cpd(self):
		"""
		Parameters
		----------
		None

		Return
		------
		Conditional Probability Distribution

		Examples
		--------

		>>> x = Node('X',parents=['Y','Z'])
		>>> x.add_cpd([[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3],
		 									[0.3, 0.7, 0.02, 0.2]], [2,2])
		>>> x.get_cpd()


		"""
		return self.cpd

	def get_edges(self):
		"""
		Parameters
		----------
		None

		Return
		------
		List of edges in the form (parent,child)

		Examples
		--------

		>>> x = Node('X',parents=['Y','Z'])
		>>> x.get_edges()
		[('Y','X'),('Z','X')]

		"""
		edges = []
		for parent in self.parents:
			edges.append((parent,self.nodeName))
		return edges

	def get_name(self):
		"""
		Parameters
		----------
		None

		Return
		------
		String Name

		Examples
		--------

		>>> g = Node('Grade',parents=['Difficulty','Intelligence'])
		>>> g.get_name()
		'Grade'

		"""
		return self.nodeName
		
	def define_structure(self):

		self.model.add_edge(['pageCategory','pagePopularity'],types=['d','lgandd'])
		self.model.add_edge(['pagePopularity','pageTalkingAbout'],types=['lgandd','lg'])
		self.model.add_edge(['pageTalkingAbout','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['postPromotion','Comments'],types=['d','lgandd'])
		self.model.add_edge(['postLength','postShareCt'],types=['lg','lg'])
		self.model.add_edge(['postLength','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['postShareCt','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['baseTime','cc1'],types=['lg','lg'])
		self.model.add_edge(['baseDay','cc2'],types=['d','lgandd'])
		self.model.add_edge(['cc1','cc2'],types=['lg','lgandd'])
		self.model.add_edge(['cc2','cc3'],types=['lgandd','lg'])
		self.model.add_edge(['cc3','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['pageCheckins','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['postDay','cc4'],types=['d','lgandd'])
		self.model.add_edge(['cc4','Comments'],types=['lgandd','lgandd'])


		#g = self.ndata#Ndata(self.fname)
		
		#print self.model.get_parents("pagePopularity")
		
		DISCRETE = "d"
		LINEARGAUSSIAN = "lg"
		LGANDDISCRETE = "lgandd"
		
		#print self.get_value("pageCategory", DISCRETE)
		
		dat = {"pageCategory": self.get_value("pageCategory", DISCRETE), 
		"pagePopularity": self.get_value("pagePopularity", LGANDDISCRETE), 
		"pageTalkingAbout": self.get_value("pageTalkingAbout", LINEARGAUSSIAN), 
		"postPromotion": self.get_value("postPromotion", LINEARGAUSSIAN),
		"postLength": self.get_value("postLength", LINEARGAUSSIAN),
		"postShareCt": self.get_value("postShareCt", LINEARGAUSSIAN),
		"baseDay": self.get_value("baseDay", DISCRETE),
		"cc1": self.get_value("cc1", LINEARGAUSSIAN),
		"cc2": self.get_value("cc2", LGANDDISCRETE),
		"cc3": self.get_value("cc3", LINEARGAUSSIAN),
		"postDay": self.get_value("postDay", DISCRETE),
		"pageCheckins": self.get_value("pageCheckins", LINEARGAUSSIAN),
		"cc4": self.get_value("cc4", LGANDDISCRETE),
		"Comments": self.get_value("Comments", LINEARGAUSSIAN),
		}
		
		
		node_data = {"Vdata": dat}
		
		print "dict: ",node_data
		
	
	def infer(self):
		#infr = Mplp(self.model)
		f = open('query.txt', 'r')
		lines = f.readlines()
		for i in lines:
			lst = i.split(",")
			queryType = lst[0]
			if(queryType == 0):
				child = lst[1]
				parents = lst[2:]

			#print infr.query(child, parents)

		#TODO: Add handling of multiple types of queries defined in query file

		
		

if __name__=="__main__":

	fname = []
	dirname = "Training/"
	for files in os.listdir(dirname):
		if(files.endswith(".csv")):

			fname.append(dirname + files)

	#ob = Data(fname)
	bn = BNetwork(fname)
	bn.define_structure()

	#dat = pd.DataFrame(ob.data, columns = ['pagePopularity', 'pageCheckins', 'pageTalkingAbt',  'pageCategory', 'cc1', 'cc2', 'cc3', 'cc4', 'cc5','baseTime', 'postLength','postShareCt', 'postPromotion', 'hLocal', 'postDay' , 'Comments'])
	
	bn.infer()
	
	
