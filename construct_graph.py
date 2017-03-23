import numpy as np
from sklearn import linear_model


class Ndata(Data):
	
	def __init__(self, fname):
	
		Data.__init__(fname)
		
		
		
		
	def linear_regr(self, nodeValue, parents):
		
		reg = linear_model.LinearRegression()
		reg.fit(parents, nodeValue)
		return reg.coef_

	
		
	def get_value(self, nodeName, val_type):
		
		var_index = self.dictIdx[nodeName]
		node_val = self.dictVal[nodeName]
		parents = self.get_parent(nodeName)
		children = self.get_children(nodeName)
		
		
		if(val_type == "lgandd"):
			
			discrete_parents = [x for x in self.get_parent(node_val) where type == "discrete"]
			values, count = np.unique(self.get_parent(discrete_parents), return_counts = True)
			vals = []
			variance = []
			mean_base = []
			mean_scale = []
			ret = {}
			for i in vals_pageCategory:
				mask = np.zeros(len(self.data[0,:]), dtype = bool)
				mask[var_index] = True
				filter_val = [node_val[:, var_index] == i][:, mask]
				
				variance.append(np.var(filter_val))
				mean_base.append(np.mean(filter_val))
				vals.append("[%s]" % i)
				mean_scale.append(1)
				#mean_scale.append()
		
			d = {}
			for i in range(0, len(vals)):
				d[vals[i]].append({"mean_base":mean_base[i], "variance": variance[i], "mean_scal":mean_scale[i]})

		
			ret.append({"type":val_types, "parents":parents, "children":children, "hybcprob": d})

		elif(val_type == "discrete"):
			
			values, count = np.unique(node_val, return_counts = True)
			total = len(self.data[:,0])
			prob = []
			for i in values:
				mask = np.zeros(len(self.data[0,:]), dtype = bool)
				mask[var_index] = True
				filter_val = [node_val[:, var_index] == i][:, mask]
				
				prob.append(len(filter_val)/total)
							
			
			#ret = {"type": "discrete","parents": parents, "children": children, "numoutcomes": count, "cprob": [x for x in prob], "vals":[x for x in values]}
			ret.append({"type": "discrete","parents": parents, "children": children, "numoutcomes": count, "cprob": [x for x in prob], "vals":[x for x in values]})

		else:   #linear gaussian types
			
			
			beta = self.linear_regr(node_val, parents)
			
			mean_scal = beta[1:]
			mean_base = beta[0]
			
			ret.append({"type":val_types, "parents":parents, "children":children, "mean_base": mean_base, "mean_scal":[x for x in mean_scal]})
			
		return ret
		
		
