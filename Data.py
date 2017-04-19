##########################################################################################################
#
#
#	Date			Name	 		Description
#	24-Mar-2017		Anurag Dixit	Added the file for multiple inheritance across multiple classes
#	24-Mar-2017 	Anurag Dixit	Changes for logs
#	24-Mar-2017		Anurag Dixit	Bug fixes which affected column index numbers
#
#
########################################################################################################


import numpy as np
from sklearn import linear_model
from Structure import Structure
import os
import csv


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

		self.postDayIdx = 39
		self.postSunIdx = 39
		self.postMonIdx = 40
		self.postTueIdx = 41
		self.postWedIdx = 42
		self.postThuIdx = 43
		self.postFriIdx = 44
		self.postSatIdx = 45

		self.baseDayIdx = 46

		self.baseSunIdx = 46
		self.baseMonIdx = 47
		self.baseTueIdx = 48
		self.baseWedIdx = 49
		self.baseThuIdx = 50
		self.baseFriIdx = 51
		self.baseSatIdx = 52

		self.targetIdx = 53


	def __init__(self, fname):

		#self.model = Structure()
		self.consideredVars = [0, 1 , 2, 3, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 46, 47, 48, 49, 50, 51, 52, 53]

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
#		self.model = LinearGaussianBayesianNetwork()

	def initialize_vars(self):

		self.pagePopularity = self.data[:,self.pagePopularityIdx]
		self.pageCheckins = self.data[:,self.pageCheckinsIdx]
		self.pageTalkingAbt = self.data[:,self.pageTalkingAbtIdx]
		self.pageCategory = self.data[:,self.pageCategoryIdx]
		self.cc1 = self.data[:,self.cc1Idx]
		self.cc2 = self.data[:,self.cc2Idx]
		self.cc3 = self.data[:,self.cc3Idx]
		self.cc4 = self.data[:,self.cc4Idx]
		self.cc5 = self.data[:,self.cc5Idx]
		self.baseTime = self.data[:,self.baseTimeIdx]
		self.postLength = self.data[:,self.postLengthIdx]
		self.postShareCt = self.data[:,self.postShareCtIdx]
		self.postPromotion = self.data[:,self.postPromotionIdx]

		self.hLocal = self.data[:,self.hLocalIdx]
		self.postSun = self.data[:,self.postSunIdx]
		self.postMon = self.data[:,self.postMonIdx]
		self.postTue = self.data[:,self.postTueIdx]
		self.postWed = self.data[:,self.postWedIdx]
		self.postThu = self.data[:,self.postThuIdx]
		self.postFri = self.data[:,self.postFriIdx]
		self.postSat = self.data[:,self.postSatIdx]


		self.baseSun = self.data[:,self.baseSunIdx]
		self.baseMon = self.data[:,self.baseMonIdx]
		self.baseTue = self.data[:,self.baseTueIdx]
		self.baseWed = self.data[:,self.baseWedIdx]
		self.baseThu = self.data[:,self.baseThuIdx]
		self.baseFri = self.data[:,self.baseFriIdx]
		self.baseSat = self.data[:,self.baseSatIdx]



		self.target = self.data[:,self.targetIdx]




		postPubDays = np.zeros((7, len(self.postSun)))
		baseDays = np.zeros((7, len(self.baseSun)))


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

		baseDays[0] = self.baseSun
		baseDays[1] = self.baseMon
		baseDays[2] = self.baseTue
		baseDays[3] = self.baseWed
		baseDays[4] = self.baseThu
		baseDays[5] = self.baseFri
		baseDays[6] = self.baseSat

		self.baseDay = np.array(np.argmax(baseDays,
							axis=0).reshape(len(self.baseSun),1),
							dtype=np.float32)


		#self.postDay = self.reduceDimension(postPubDays)

		self.data[:,self.postSunIdx] = self.postDay[:,0]
		self.data[:,self.baseSunIdx] = self.baseDay[:,0]

		completeList = []
		for i in range(0, 54):
			completeList.append(i)
		trun = np.setdiff1d(completeList, self.consideredVars)

		# Need to reverse as the col indexes will change on deletion of columns
		trun = trun[::-1]

		print "Dimensionality Reduction Stage ... "
		print "Filtering out variables : ", trun
		for i in range(0, len(trun)):

			self.data = np.delete(self.data, trun[i], 1)


		keys = ['pagePopularity', 'pageCheckins','pageTalkingAbout','pageCategory','cc1','cc2','cc3','cc4','cc5','baseTime','postLength','postShareCt','postPromotion','hLocal','postDay','baseDay','Comments']
		idx = range(len(keys))
		self.dictVal = {'pagePopularity':self.pagePopularity, 'pageCheckins':self.pageCheckins, 'pageTalkingAbout':self.pageTalkingAbt, 'pageCategory':self.pageCategory, 'cc1':self.cc1, 'cc2':self.cc2, 'cc3':self.cc3, 'cc4':self.cc4, 'cc5':self.cc5, 'baseTime':self.baseTime, 'postLength':self.postLength, 'postShareCt':self.postShareCt, 'postPromotion':self.postPromotion, 'hLocal':self.hLocal, 'postDay':self.postDay, 'baseDay':self.baseDay ,'Comments':self.target }
		#self.dictIdx = {'pagePopularity':self.pagePopularityIdx, 'pageCheckins':self.pageCheckinsIdx, 'pageTalkingAbout':self.pageTalkingAbtIdx, 'pageCategory':self.pageCategoryIdx, 'cc1':self.cc1Idx, 'cc2':self.cc2Idx, 'cc3':self.cc3Idx, 'cc4':self.cc4Idx, 'cc5':self.cc5Idx, 'baseTime':self.baseTimeIdx, 'postLength':self.postLengthIdx, 'postShareCt':self.postShareCtIdx, 'postPromotion':self.postPromotionIdx, 'hLocal':self.hLocalIdx, 'postDay':self.postDayIdx, 'baseDay': self.baseDayIdx, 'Comments':self.targetIdx }
		self.dictIdx = dict(zip(keys, idx))

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
