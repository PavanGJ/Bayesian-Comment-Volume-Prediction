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
	

	def initialize_indexes(self):

		self.pagePopularityIdx = 0
		self.pageCheckinsIdx = 1
		self.pageTalkingAbtIdx = 2
		self.pageCategoryIdx = 3
		self.cc1Idx = 29
		self.cc2Idx = 30
		self.cc3Idx = 31
		self.cc4Idx = 33
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

		self.initialize_indexes()
		self.root = 0
		lst = []	
		#self.pos_dict = {'pagePopularity':0, 'pageCheckins':1, 'pageTalkingAbout':2, 'pageCategory':3, 'baseTime':34, 'postLength':35, 'postShareCount':36, 'postStatusPromotion':37, 'postDay':[39, 40, 41, 42, 43, 44, 45], 'comment':53}

		self.category = {}
		#self.pos = [0, 1, 2, 3, 34, 35, 36, 37, [39, 40, 41, 42, 43, 44, 45], 53]	
		for f in fname:
			#print "andi ",f
			with open(f) as fil:
				reader = csv.DictReader(fil)
				for row in reader:
					lst.append(row)		
		
		self.data = np.array(lst)		
		#print self.data
		
		self.initialize_vars()
	
	def initialize_vars(self):

		self.pagePopularity = self.data[self.pagePopularityIdx]
		self.pageCheckins = self.data[self.pageCheckinsIdx]
		self.pageTalkingAbt = self.data[self.pageTalkingAbtIdx]
		self.pageCategory = self.data[self.pageCategoryIdx]
		self.cc1 = self.data[self.cc1Idx]
		self.cc2 = self.data[self.cc2Idx]
		self.cc3 = self.data[self.cc3Idx]
		self.cc4 = self.data[self.cc4Idx]
		self.baseTime = self.data[self.baseTimeIdx]
		self.postLength = self.data[self.postLengthIdx]
		self.postShareCt = self.data[self.postShareCtIdx]
		self.postPromotion = self.data[self.postPromotionIdx]
		self.hLocal = self.data[self.hLocalIdx]
		self.postSun = self.data[self.postSunIdx]
		self.postMon = self.data[self.postMonIdx]
		self.postTue = self.data[self.postTueIdx]
		self.postWed = self.data[self.postWedIdx]
		self.postThu = self.data[self.postThuIdx]
		self.postFri = self.data[self.postFriIdx]
		self.postSat = self.data[self.postSatIdx]
		self.target = self.data[self.targetIdx]

		postPubDays = [self.postSun, self.postMon, self.postTue, self.postWed, self.postThu, self.postFri, self.postSat]
		
		self.reduceDimension(postPubDays)
		
		#print self.pagePopularity
	
	def reduceDimension(self, lst):
		
		postDays = np.zeros(len(lst[0]))
		#for i in range(0,len(lst[0])):
			

	

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
			
				
	def define_structure(self):
		
		#self.root = ret_val
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
	
	
