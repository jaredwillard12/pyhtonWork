import numpy as np

class COCOData(object):

	def __init__(self, pathToData='/home/willarjc/COCOdata/annotations/instances_train2014.json', typeOfData='COCO', batchsize=256, vbatchsize=256):
		import json
		self.jf = json.load(open(pathToData))
		self.typeOfData = typeOfData
		self.batchsize = batchsize
		self.validationBatchsize = vbatchsize
		self.name = None
		self.idNum = None
		self.data = []
		self.labels = []
		self.validationX = []
		self.validationY = []
		self.pureData = []
		self.catToImgs = None

#loads pure data by id name, and loads a set that contains photos that contain both objects
	def loadPureDataByIdName(self, idNameA, idNameB=None):
		self.pureData = []
		if self.catToImgs is None:
			self.getImgsByCat()

		max=len(self.jf['categories'])
		i=0
		while self.jf['categories'][i]['name']!=idNameA:
			i+=1
			if i==max:                
				return 0
		idNumA=self.jf['categories'][i]['id']

		if idNameB!=None:
			i = 0
			while self.jf['categories'][i]['name']!=idNameB:
				i+=1
				if i==max:
					return 0
			idNumB=self.jf['categories'][i]['id']

		numExtension = '000000000000'
		path = '/home/willarjc/COCOdata/train2014/COCO_train2014_'
		fullPathName = lambda x: path + numExtension[0:-len(str(x))] + str(x) + ".jpg"
		if idNameB!=None:
			intersection = list(set(self.catToImgs[idNumA]) & set(self.catToImgs[idNumB]))
			isolatedA = list(set(self.catToImgs[idNumA])-set(intersection))
			isolatedB = list(set(self.catToImgs[idNumB])-set(intersection))
			self.pureData.append([fullPathName(en) for en in isolatedA])
			self.pureData.append([fullPathName(en) for en in isolatedB])
			self.pureData.append([fullPathName(en) for en in intersection])
		else:
			isolatedA = list(set(self.catToImgs[idNumA]))
			self.pureData.append([fullPathName(en) for en in isolatedA])

#catToImgs contains all the img ids associated with a certain category
	def getImgsByCat(self):
		catToImgs={}
		if 'annotations' in self.jf and 'categories' in self.jf:
			for ann in self.jf['annotations']:
				try:
					catToImgs[ann['category_id']].append(ann['image_id'])
				except KeyError:
					catToImgs[ann['category_id']]=[]
		self.catToImgs = catToImgs

#returns the intersection between two idNums (list of img ids)
	def getOverlap(self, idNumA, idNumB):
		if self.catToImgs is None:
			self.getImgsByCat()
		return list(set(self.catToImgs[idNumA]) & set(self.catToImgs[idNumB]))

	def dataLoadAnythingElse(self):
		elseData=[]
		elseLabels=[]
		numExtension = '000000000000'
		path = '/home/willarjc/COCOdata/train2014/COCO_train2014_'
		fullPathName = lambda x: path + numExtension[0:-len(str(x))] + str(x) + ".jpg"
		randIsTrue = np.random.randint(2, size=100)
		count=0
		for en in self.jf['annotations']:
			catId = en['category_id']
			imageID = en['image_id']
			if catId==self.idNum:
				continue
			elif randIsTrue[count%100]:
				elseData.append(fullPathName(imageID))
				elseLabels.append(np.array([1,0]))
			count+=1
		return elseData, elseLabels

	def dataLoadByNumber(self, idNum, fromName=False):
		self.data=[]
		self.labels=[]
		self.validationX = []
		self.validationY = []
		self.pureData = []
		numExtension = '000000000000'
		path = '/home/willarjc/COCOdata/train2014/COCO_train2014_'
		fullPathName = lambda x: path + numExtension[0:-len(str(x))] + str(x) + ".jpg"
		i=0
		if fromName:
			i=idNum
			idNum=self.jf['categories'][i]['id']
		max=len(self.jf['categories'])
		while self.jf['categories'][i]['id']!=idNum:
			i+=1
			if i==max:
				return 0
		self.name=self.jf['categories'][i]['name']
		self.idNum=idNum
		count=0
		for en in self.jf['annotations']:
			catId=en['category_id']
			imageID = en['image_id']
			if catId!=idNum:
				continue
			if count%3==0:
				self.validationX.append(fullPathName(imageID))
				self.validationY.append(np.array((0,1)))
			else:
				self.labels.append(np.array((0,1)))
				self.data.append(fullPathName(imageID))
			count+=1
		self.pureData = self.data
		overallCount=0
		for en in self.jf['annotations']:
			catId=en['category_id']
			imageID = en['image_id']
			if catId!=idNum and count>0 and overallCount%25==0:
				if count%3==0:
					self.validationX.append(fullPathName(imageID))
					self.validationY.append(np.array((1,0)))
				else:
					self.labels.append(np.array((1,0)))
					self.data.append(fullPathName(imageID))
				count-=1
			overallCount+=1
		return 1
			
	def dataLoadByName(self, name):
		i=0
		if name=='anything':
			self.dataLoadAnything()
		max=len(self.jf['categories'])
		while self.jf['categories'][i]['name']!=name:
			i+=1
			if i==max:
				return 0
		self.dataLoadByNumber(i, fromName=True)
		return 1

	def nextBatch(self, validation=False):
		import random
		if len(self.data)==0:
			print("There is no data")
			return 0
		if not validation:
			randomSample = random.sample(range(len(self.data)), self.batchsize)
			samplingData = [self.data[i] for i in randomSample]
			samplingLabels = [self.labels[i] for i in randomSample]
		if validation:
			randomSample = random.sample(range(len(self.validationX)), self.validationBatchsize)
			samplingData = [self.validationX[i] for i in randomSample]
			samplingLabels = [self.validationY[i] for i in randomSample]
		return samplingData, samplingLabels
