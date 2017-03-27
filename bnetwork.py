################################################################################
#
#	Date 			Name			Description
#
#	01-Mar-2017     Anurag Dixit	Initial Draft
#	19-Mar-2017		Anurag Dixit	Added API for conditional probabilities
#	20-Mar-2017  	Anurag Dixit	Bug fix for parsing of data correctly
#	20-Mar-2017		Anurag Dixit	Added changes for Bayesian Model incorporation and Data
#	20-Mar-2017		Anurag Dixit	Added file read for query perform and commented the MPLP
#	21-Mar-2017		Pavan Joshi		Depreciated reduceDimensions function to utilize numpy functions
#	21-Mar-2017		Pavan Joshi		Added API to handle nodes in the network.
#	22-Mar-2017		Anurag Dixit	Added Linear Regression Code for adding cpds for continuos variables
#	24-Mar-2017     Anurag Dixit    Changes done for hybrid bayesian network model compatible data generation
#	24-Mar-2017     Anurag Dixit	Changes for Linear Regression intercept
#	24-Mar-2017		Pavan Joshi		Adding API to Create a Hybrid Bayesian Network using libpgm
#	25-Mar-2017     Anurag Dixit	Added API for calculation of entropy and KL Divergence
#	26-Mar-2017 	Pavan Joshi		Added API to get independencies in the model
#	27-Mar-2017		Pavan Joshi		Added API to handle inference queries and evaluation metrics
#
################################################################################
import os
import csv
import json
import numpy as np


from pgmpy.models import BayesianModel
from libpgm.hybayesiannetwork import HyBayesianNetwork
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.sampleaggregator import SampleAggregator
from sklearn import linear_model

from construct_graph import Ndata
from Structure import Structure
from Data import Data
import scipy.stats

class BNetwork(Ndata):


	def __init__(self, fname):

		Ndata.__init__(self, fname)
		self.fname = fname
		self.model = Structure()
		self.independencies_model = BayesianModel()


	def define_structure(self):

		print "Constructing the Hybrid Bayesian Network Model graph ... "

		self.model.add_edge(['pageCategory','pagePopularity'],types=['d','lgandd'])
		self.model.add_edge(['pagePopularity','pageTalkingAbout'],types=['lgandd','lg'])
		self.model.add_edge(['pageTalkingAbout','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['postPromotion','Comments'],types=['d','lgandd'])
		self.model.add_edge(['postLength','postShareCt'],types=['lg','lg'])
		self.model.add_edge(['postLength','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['postShareCt','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['baseDay','cc2'],types=['d','lgandd'])
		self.model.add_edge(['cc1','cc2'],types=['lg','lgandd'])
		self.model.add_edge(['cc2','cc3'],types=['lgandd','lg'])
		self.model.add_edge(['cc3','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['pageCheckins','Comments'],types=['lg','lgandd'])
		self.model.add_edge(['postDay','cc4'],types=['d','lgandd'])
		self.model.add_edge(['cc4','Comments'],types=['lgandd','lgandd'])

		self.independencies_model.add_edge('pageCategory','pagePopularity')
		self.independencies_model.add_edge('pagePopularity','pageTalkingAbout')
		self.independencies_model.add_edge('pageTalkingAbout','Comments')
		self.independencies_model.add_edge('postPromotion','Comments')
		self.independencies_model.add_edge('postLength','postShareCt')
		self.independencies_model.add_edge('postLength','Comments')
		self.independencies_model.add_edge('postShareCt','Comments')
		self.independencies_model.add_edge('baseDay','cc2')
		self.independencies_model.add_edge('cc1','cc2')
		self.independencies_model.add_edge('cc2','cc3')
		self.independencies_model.add_edge('cc3','Comments')
		self.independencies_model.add_edge('pageCheckins','Comments')
		self.independencies_model.add_edge('postDay','cc4')
		self.independencies_model.add_edge('cc4','Comments')

		with open("structure.json","wb") as json_file:
			json_file.write(self.model.get_structure())



		DISCRETE = "d"
		LINEARGAUSSIAN = "lg"
		LGANDDISCRETE = "lgandd"


		print "Calculating CPDs compatible to Hybrid Bayesian Network Model for Hybrid Data ... "

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


		self.node_data = {"Vdata": dat}

		with open("nodedata.json","wb") as json_file:
			json_file.write(json.dumps(self.node_data, indent=2))

	def get_independencies(self,variables=None):
		if variables == None:
			return self.independencies_model.get_independencies()
		return self.independencies_model.local_independencies(variables)

	def probability_query(self,query,evidence = {}):
		prob = 1
		if len(evidence.keys()) == 0:
			samples = self.bayesian_network.randomsample(1000)
		else:
			samples = self.bayesian_network.randomsample(1000,evidence)
		aggregate = self.aggregator.aggregate(samples)
		for key in query.keys():
			valDict = aggregate[key]
			if(self.model.get_vertex_type(key)!="d"):
				values = np.array(valDict.keys(),dtype=np.float)
				dist = scipy.stats.norm(np.mean(values),np.var(values))
				prob *= dist.pdf(query[key])
			else:
				prob *= float(valDict[str(query[key])])
		return prob


	def infer(self):
		f = open('query.txt', 'r')
		lines = f.readlines()
		for i in lines:
			lst = i.strip().split(", ")
			queryType = lst[0]
			if(queryType == 'I'):
				print "\n\n##########    Printing all independencies    ##########\n"
				print self.get_independencies()
			elif(queryType == 'LI'):
				var = lst[1].strip("[").strip("]").split(" & ")
				print "\n\n#########    Local independencies for",var,"   #########\n"
				print self.get_independencies(variables = var)
			elif(queryType == 'CP'):
				evidences = dict()
				query = dict()
				args = lst[1].split("=")
				query[args[0].strip().strip(" ")] = float(args[1].strip().strip(" "))
				args = lst[2].split(" -> ")
				for arg in args[1].strip("[").strip("]").split("&"):
					evid = arg.strip().strip(" ").split("=")
					evidences[evid[0].strip().strip(" ")] = float(evid[1].strip().strip(" "))
				print "\n\n##########    P(",lst[1],"|",lst[2].split(" -> ")[1],")    ##########\n"
				print self.probability_query(query,evidences)
			elif(queryType == 'M'):
				query = dict()
				args = lst[1].split("=")
				query[args[0].strip().strip(" ")] = float(args[1].strip().strip(" "))
				print "\n\n##########    P(",lst[1],")    ##########\n"
				print self.probability_query(query)

		#TODO: Add handling of multiple types of queries defined in query file

	def metric(self, a, b):

		#Make sure the input parameters are probability Distributions

		entropy = scipy.stats.entropy(a,base=10)
		kl_divergence = scipy.stats.entropy(a, b)
		return entropy, kl_divergence



	def create_network(self):
		skeleton = GraphSkeleton()
		skeleton.load("structure.json")
		ndata = NodeData()
		ndata.load("nodedata.json")
		skeleton.toporder()
		ndata.entriestoinstances()

		self.bayesian_network = HyBayesianNetwork(skeleton,ndata)
		self.aggregator = SampleAggregator()

	def evaluation_metrics(self):
		result = self.bayesian_network.randomsample(1000)
		aggregate = self.aggregator.aggregate(result)['Comments']
		samples = np.array(aggregate.keys(),dtype=np.float)
		samplepdf = scipy.stats.norm(np.mean(samples),np.var(samples))
		origin_data = np.array(self.target,dtype=np.float)
		originalpdf = scipy.stats.norm(np.mean(origin_data),np.var(origin_data))
		query_domain = np.linspace(np.mean(origin_data) - np.var(origin_data),
		 						np.mean(origin_data) + np.var(origin_data), 100)
		entropy, kl_divergence = self.metric(samplepdf.pdf(query_domain),originalpdf.pdf(query_domain))
		print "\n\n##########    Evaluation Metrics    ##########\n"
		print "Entropy:",entropy
		print "KL Divergence:",kl_divergence


if __name__=="__main__":

	fname = []
	dirname = "Training/"
	for files in os.listdir(dirname):
		if(files.endswith(".csv")):

			fname.append(dirname + files)

	bn = BNetwork(fname)
	bn.define_structure()
	bn.create_network()
	bn.infer()
	bn.evaluation_metrics()
