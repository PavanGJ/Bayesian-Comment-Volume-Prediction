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
#
################################################################################
import os
import csv
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import Mplp
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD

class Node:

	def __init__(self,nodeName,parents = [],continuous = False):
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


class Data:

	def initialize_indexes(self):

		self.pagePopularityIdx = 0
		self.pageCheckinsIdx = 1
		self.pageTalkingAbtIdx = 2
		self.pageCategoryIdx = 3
		self.cc1Idx = 29
		self.cc2Idx = 30
		self.cc3Idx = 31
		self.cc4Idx = 32
		self.cc5Idx = 33
		self.baseTimeIdx = 34
		self.postLengthIdx = 35
		self.postShareCtIdx = 36
		self.postPromotionIdx = 37
		self.hLocalIdx = 38
		self.postSunIdx = 39
		self.postMonIdx = 40
		self.postTueIdx = 41
		self.postWedIdx = 42
		self.postThuIdx = 43
		self.postFriIdx = 44
		self.postSatIdx = 45
		self.targetIdx = 53


	def __init__(self, fname):

		self.consideredVars = [0, 1 , 2, 3, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 53]

		self.NUM_VAR = 15
		self.initialize_indexes()
		self.root = 0
		lst = []

		for f in fname:
			with open(f) as fil:

				reader = csv.reader(fil, delimiter=',', quoting=csv.QUOTE_NONE)
				for row in reader:

					lst.append(row)

		self.data = np.array(lst)

		self.initialize_vars()
		self.model = BayesianModel()

	def initialize_vars(self):

		self.pagePopularity = self.data[self.pagePopularityIdx]
		self.pageCheckins = self.data[self.pageCheckinsIdx]
		self.pageTalkingAbt = self.data[self.pageTalkingAbtIdx]
		self.pageCategory = self.data[self.pageCategoryIdx]
		self.cc1 = self.data[self.cc1Idx]
		self.cc2 = self.data[self.cc2Idx]
		self.cc3 = self.data[self.cc3Idx]
		self.cc4 = self.data[self.cc4Idx]
		self.cc5 = self.data[self.cc5Idx]
		self.baseTime = self.data[self.baseTimeIdx]
		self.postLength = self.data[self.postLengthIdx]
		self.postShareCt = self.data[self.postShareCtIdx]
		self.postPromotion = self.data[self.postPromotionIdx]
		self.hLocal = self.data[self.hLocalIdx]
		self.postSun = self.data[:,self.postSunIdx]
		self.postMon = self.data[:,self.postMonIdx]
		self.postTue = self.data[:,self.postTueIdx]
		self.postWed = self.data[:,self.postWedIdx]
		self.postThu = self.data[:,self.postThuIdx]
		self.postFri = self.data[:,self.postFriIdx]
		self.postSat = self.data[:,self.postSatIdx]
		self.target = self.data[self.targetIdx]


		postPubDays = np.zeros((7, len(self.postSun)))

		postPubDays[0] = self.postSun
		postPubDays[1] = self.postMon
		postPubDays[2] = self.postTue
		postPubDays[3] = self.postWed
		postPubDays[4] = self.postThu
		postPubDays[5] = self.postFri
		postPubDays[6] = self.postSat
		self.postDay = np.array(np.argmax(postPubDays,
							axis=0).reshape(len(self.postSun),1),
							dtype=np.float32)

		#self.postDay = self.reduceDimension(postPubDays)

		self.data[:,self.postSunIdx] = self.postDay[:,0]

		completeList = []
		for i in range(0, 54):
			completeList.append(i)
		trun = np.setdiff1d(completeList, self.consideredVars)

		# Need to reverse as the col indexes will change on deletion of columns
		trun = trun[::-1]

		print "Filtering out variables : ", trun
		for i in range(0, len(trun)):

			self.data = np.delete(self.data, trun[i], 1)


################################################################################
#
#			Depreciated to utilize optimized functionalities of numpy
#
################################################################################
#
#	def reduceDimension(self, lst):
#
#		postDay = np.zeros((len(lst[0]), 1))
#		for i in range(0,len(lst[0])):
#			postDay[i] = np.argmax(lst[:,i])
#
#		return postDay
#
################################################################################

	def getCondProb(self, child, parents, Y):

		N = len(child)
		k = len(parents)
		y = np.zeros((k+1))

		x0 = np.ones((N + 1))
		A = np.zeros((k+1, k+1))
		X = np.vstack((x0,parents))

		for i in range(0, len(parents)):
			for j in range(0, len(parents)):

				A[i][j] = np.dot(X[j],X[i])

		print "A ",A


		#Calculating y
		for i in range(0, k+1):
			y[i] = np.dot(Y, X[i])

		#Calculating Beta

		beta = np.dot(np.linalg.pinv(A), y)

		#calculating variance
		sum_val = 0
		for i in range(0, k+1):
			sum_val = sum_val + np.dot(beta[i],X[i])

		sum_val = sum_val - y
		var = pow(sum_val, 2)/N

		#Now that we got all the ingredients lets calculate the conditional probabilities

		logval = -1/2 * np.log(2*np.pi*var) - 1/(2*var) * pow(sum_val, 2)


		return pow(e, logval)

################################################################################
#
#	Setting this function up for changes according to the new Node class
#
################################################################################
	def define_structure(self):

		self.model.add_edges_from([('pageCategory','pagePopularity'),('pagePopularity', 'pageTalkingAbt')])
		self.model.add_edge('pageTalkingAbt', 'Comments')
		self.model.add_edge('postPromotion','Comments')
		self.model.add_edge('postLength', 'Comments')
		self.model.add_edges_from([('postLength', 'postShareCt'), ('postShareCt','Comments')])
		self.model.add_edges_from([('baseDay','cc2'),('cc2', 'cc3')])
		self.model.add_edge('cc3', 'Comments')
		self.model.add_edge('pageCheckins','Comments')
		self.model.add_edges_from([('baseTime','cc1'), ('cc1', 'cc2')])
		self.model.add_edges_from([('postDay','cc4'),('cc4', 'Comments')])
		self.model.add_edge('hLocal','Comments')



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

	ob = Data(fname)
	ob.define_structure()

	dat = pd.DataFrame(ob.data, columns = ['pagePopularity', 'pageCheckins', 'pageTalkingAbt',  'pageCategory',
	'cc1', 'cc2', 'cc3', 'cc4', 'cc5','baseTime', 'postLength','postShareCt', 'postPromotion', 'hLocal', 'postDay' , 'Comments'])
	#ob.model.fit(dat)
	ob.infer()
