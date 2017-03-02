###################################################################################
#
#	Date 		Name		Description
#
#	01-Mar-2017     Anuag Dixit	Initial Draft
#
################################################################################
import os
import csv
import numpy as np

class Data:
	
	def __init__(self, fname):
		'''
		self.pagePopularityIndex = 0
		self.pageChecinsIndex = 1
		self.pageTalkingAbtIndex = 2
		self.pageCategoryIndex = 3
		self.baseTimeIndex = 34
		self.postLengthIndexIndex = 35
		self.postShareCountIndex = 36
		self.postStatusPromotionIndex = 37
		self.postDay = [39, 40, 41, 42, 43, 44, 45]	
		self.commentIndex = 53
		'''
		self.root = 0
		lst = []	
		self.pos_dict = {'pagePopularity':0, 'pageCheckins':1, 'pageTalkingAbout':2, 'pageCategory':3, 'baseTime':34, 'postLength':35, 'postShareCount':36, 'postStatusPromotion':37, 'postDay':[39, 40, 41, 42, 43, 44, 45], 'comment':53}

		self.category = {}
		#self.pos = [0, 1, 2, 3, 34, 35, 36, 37, [39, 40, 41, 42, 43, 44, 45], 53]	
		for f in fname:
			#print "andi ",f
			with open(f) as fil:
				reader = csv.DictReader(fil)
				for row in reader:
					lst.append(row)		
		
		self.data = np.array(lst)		
		print self.data
		
	def preParseDate(self):
		
		pass

	def define_structure(self):
		
	
		self.root = ret_val
		pass
	
	def dimensionality_reduction(self):
		pass
	
	def infer(self):
		
		pass


if __name__=="__main__":
	
	fname = []
	dirname = "Training/"
	for files in os.listdir(dirname):
		if(files.endswith(".csv")):
			fname.append(dirname + files)
	
	ob = Data(fname)
	#print fname
	
	
