from __future__ import division
#############################################################################################################
#
#
#
#	Date 				Name	 		Description
#	24-Mar-2017			Anurag Dixit		Added the file for creating the data for hybridBayesianNetwork Model
#	24-Mar-2017			Anurag Dixit		Bug fix (mean_scale for Comments)
#	24-Mar-2017			Pavan Joshi		Minor Bug fix to overcome indexing problems in some systems
#	25-Mar-2017			Pavan Joshi		Minor fixes to overcome bugs when serializing numpy array and values
#	26-Mar-2017			Pavan Joshi		Bug fixes for creation of dictionary that affected further processing
#	26-Mar-2017			Pavan Joshi		Added functionality to handle continuous parents for hybrid nodes
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
		return reg.coef_, reg.intercept_



	def get_value(self, nodeName, val_type):
		print "get_value", nodeName, val_type

		st = self.model #Structure()
		var_index = self.dictIdx[nodeName]
		node_val = self.dictVal[nodeName]
		parents = st.get_parents(nodeName)
		children = st.get_children(nodeName)


		if(val_type == "lgandd"):

			discrete_parents = []
			non_discrete = []
			for i in parents:
				#print i,"type",st.get_vertex_type(i)
				if(st.get_vertex_type(i) == "d"):
					discrete_parents.append(i)
				else:
					non_discrete.append(i)


			print "Discrete parents", discrete_parents
			disc_parents = self.dictVal[discrete_parents[0]]
			disc_index = self.dictIdx[discrete_parents[0]]
			values = np.array(np.unique(disc_parents),  dtype = 'float')

			count = len(values)

			vals = []
			variance = []
			mean_base = []
			mean_scale = []


			for i in values:
				filter_val = self.data[:, var_index][np.array(self.data[:, disc_index],dtype=np.float) == i]

				a = np.array(filter_val).astype(np.float)
				if len(non_discrete)==0:
					mean_base.append(float(np.mean(a)))
					mean_scale.append([])
				else:
					v = self.dictVal[non_discrete[0]]
					v = v[np.array(self.data[:, disc_index],dtype=np.float) == i]
					parent = v.reshape(len(v),1)

					for i in range(1, len(non_discrete)):
						val = self.dictVal[non_discrete[i]][np.array(self.data[:, disc_index],dtype=np.float) == i]
						parent = np.concatenate((parent, val.reshape(len(val), 1)), axis=1).astype(np.float)

					node_val = filter_val.reshape(len(filter_val), 1).astype(np.float)
					w, beta_0 = self.linear_regr(node_val, parent)

					mean_base.append(beta_0.tolist()[0])
					mean_scale.append([y for x in w.tolist() for y in x])
				vals.append("['%s']" % float(i))
				variance.append(float(np.var(a)))


			d = {}
			for i in range(0, len(vals)):

				dic = {vals[i] : {"mean_base":mean_base[i], "variance": variance[i], "mean_scal":mean_scale[i]} }
				d.update(dic)



			ret = {"type":val_type, "parents":parents, "children":children, "hybcprob": d}

			return ret

		elif(val_type == "d"):

			values = np.unique(node_val)
			count = len(values)
			total = len(self.data[:,0])

			prob = []
			for i in values:
				filter_val = np.array(self.data[np.array(self.data[:, var_index],dtype=np.float)==float(i)])
				prob.append(len(filter_val)/total)

			ret = {"type": "discrete","parents": parents, "children": children, "numoutcomes": float(count), "cprob": [float(x) for x in prob], "vals":[float(x) for x in values]}
			return ret

		else:   #linear gaussian types
			mean_scal = []
			mean_base = None

			node_val = node_val.reshape(len(node_val), 1).astype(np.float)
			variance = float(np.var(node_val))
			if(parents == None):
				mean_base = np.mean(node_val)
				#mean_scal.append(1)


			else:

				v = self.dictVal[parents[0]]
				parent = v.reshape(len(v),1)

				for i in range(1, len(parents)):
					val = self.dictVal[parents[i]]
					parent = np.concatenate((parent, val.reshape(len(val), 1)), axis=1).astype(np.float)


				w, beta_0 = self.linear_regr(node_val, parent)

				mean_base = beta_0.tolist()[0]
				mean_scal = [y for x in w.tolist() for y in x]

			ret = {"type":val_type, "parents":parents, "children":children, "mean_base": mean_base, "mean_scal":[x for x in mean_scal], "variance": variance}

			return ret
