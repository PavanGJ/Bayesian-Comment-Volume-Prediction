from __future__ import division
#############################################################################################################
#
#
#
#	Date 		Name 		Description
#	24-Mar-2017	Anurag Dixit	Added the file for creating the data for hybridBayesianNetwork Model
#
#
#
############################################################################################################

import numpy as np
from sklearn import linear_model
from Data import Data

class Ndata(Data):
	
	def __init__(self, fname):
		
		Data.__init__(self, fname)
		
		
		
		
	def linear_regr(self, nodeValue, parents):
		
		reg = linear_model.LinearRegression()
		reg.fit(parents, nodeValue)
		return reg.coef_

	
		
	def get_value(self, nodeName, val_type):
		print "get_value", nodeName, val_type
		
		st = self.model #Structure()
		var_index = self.dictIdx[nodeName]
		node_val = self.dictVal[nodeName]
		parents = st.get_parents(nodeName)
		children = st.get_children(nodeName)
		
		
		if(val_type == "lgandd"):
			
			
			#print "parents",parents
			discrete_parents = []
			for i in parents:
				#print i,"type",st.get_vertex_type(i)
				if(st.get_vertex_type(i) == "d"):
					discrete_parents.append(i)
					
			
			#discrete_parents = [x for x in pts where type == "discrete"]
			print "Discrete parents", discrete_parents
			disc_parents = self.dictVal[discrete_parents[0]]
			disc_index = self.dictIdx[discrete_parents[0]]
			values = np.array(np.unique(disc_parents),  dtype = 'float')
			
			count = len(values)
			
			vals = []
			variance = []
			mean_base = []
			mean_scale = []
			#ret = {}
			#print "values", values
			for i in values:
				mask = np.zeros(len(self.data[0,:]), dtype = bool)
				mask[var_index] = True
				
				filter_val = self.data[:, var_index][self.data[:, disc_index] == i]
				
				#print "filtered_val", filter_val
				a = np.array(filter_val).astype(np.float)
				#print "a", a, "type", type(a),a.shape, np.mean(a), a.shape
				
				mean_base.append(np.mean(a))
				vals.append("[%s]" % i)
				mean_scale.append(1)
				variance.append(np.var(a))
				
				
			d = {}
			for i in range(0, len(vals)):
				
				dic = {vals[i] : {"mean_base":mean_base[i], "variance": variance[i], "mean_scal":mean_scale[i]} } 
				d.update(dic)

		
			
			ret = {"type":val_type, "parents":parents, "children":children, "hybcprob": d}
			#print ret
			return ret

		elif(val_type == "d"):
			
			
			values = np.unique(node_val)
			count = len(values)
			total = len(self.data[:,0])
			
			prob = []
			for i in values:
				
				filter_val = np.array(self.data[self.data[:, var_index] == i])
				prob.append(len(filter_val)/total)
							
			ret = {"type": "discrete","parents": parents, "children": children, "numoutcomes": count, "cprob": [x for x in prob], "vals":[x for x in values]}
			return ret

		else:   #linear gaussian types
			mean_scal = []
			mean_base = []
			
			node_val = node_val.reshape(len(node_val), 1).astype(np.float)
			
			if(parents == None):
				mean_base.append(np.mean(node_val))
				mean_scal.append(1)
				parents = 'null'
				
			else:
				#print "parents", parents
				v = self.dictVal[parents[0]]
				parent = v.reshape(len(v),1)
				
				for i in range(1, len(parents)):
					val = self.dictVal[parents[i]]
					parent = np.concatenate((parent, val.reshape(len(val), 1)), axis=1).astype(np.float)
				
				#print node_val.shape, parent.shape
				#print "parents", parent.T, "node ", node_val.T
				
				beta = self.linear_regr(node_val, parent)
				
				if(len(beta)==1):
					mean_scal = beta[0]
					mean_base = beta[0]
				else:	
					mean_scal = beta[1:]
					mean_base = beta[0]
				
			ret = {"type":val_type, "parents":parents, "children":children, "mean_base": mean_base, "mean_scal":[x for x in mean_scal]}
			
			return ret	
			
		
