import tensorflow as tf
import numpy as np
from PIL import Image
from scipy import misc
import DataHandler as dh
import random
from sklearn.covariance import graph_lasso
from sklearn.decomposition import PCA

objectIDNames = ['pizza','dog','cat']
#objectIDNames =  ['person', 'umbrella', 'tie', 'backpack', 'handbag', 'suitcase', 'bicycle', 'motorcyle', 'bus', 'truck', 'car', 'airplane', 'train', 'boat', 'traffic light', 'stop sign', 'bench', 'fire hydrant', 'parking meter','bird','dog','sheep']#,'elephant','zebra','cat','horse','cow','bear','giraffe', 'frisbee', 'snowboard', 'kite', 'baseball glove', 'surfboard', 'tennis racket', 'skateboard', 'baseball bat', 'sports ball', 'skis']

def getDataPoint(imgPaths, weights, biases, bSize, train=False, answers=None):

	#this preprocessing replicates the caffe network results
	def preprocessImg(img):
		img = tf.reverse(img, [2, ])
		newshape = tf.stack([256, 256])
		img = tf.image.resize_images(img, newshape)
		offset = (newshape-227)/2
		img = tf.slice(img, begin=tf.to_int64(tf.stack([offset[0], offset[1],0])), size=tf.to_int64(tf.stack([227, 227, -1])))
		mean = 0#[ 104., 117., 124.]	#caffe doesn't do mean subtraction evidently
		return tf.to_float(img) - mean

	def prepareBatch(imgPaths):
		preparedBatch = np.empty([len(imgPaths), 227, 227, 3])
		count = 0
		for imgPath in imgPaths:
			img = Image.open(imgPath).resize((227,227))
			img = img.convert('RGB')
			imgarr = np.asarray(img)
			preparedBatch[count]=imgarr
			count+=1
		return preparedBatch
	
	#read in images to testing set MUST BE 227x227
	#that size is what the fully connected layer weights are sized to handle
	data = []
	data = prepareBatch(imgPaths)
	
	nclasses = 2
	learningRate = 0.001
	dropout = 0.75	

	#x is flat input array placeholder and y is flat label array placeholder
	x = tf.placeholder(tf.float32, [None,227,227,3])
	y = tf.placeholder(tf.float32, [None, nclasses])
	keepProb = tf.placeholder(tf.float32)

	#convolutional layer
	def conv(x, W, b, strideSize=1, group=1, padding='SAME'):
		if group==1:
		    x = tf.nn.conv2d(x, W, [1,strideSize,strideSize,1], padding)
		    x = tf.nn.bias_add(x, b)
		elif group==2:
		    xGroups =  tf.split(x, group, 3)
		    wGroups = tf.split(W, group, 3) 
		    outs = [tf.nn.conv2d(i, j, [1,strideSize,strideSize,1], padding) for i,j in zip(xGroups, wGroups)]
		    x = tf.concat(outs, 3)
		    x = tf.nn.bias_add(x, b)
		return tf.nn.relu(x)
	
	#max pool layer
	def maxpoolnn(x, k=3):
		return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k-1,k-1,1], padding='VALID')
	
	#normalization layer
	def norm(x):
		return tf.nn.lrn(x, 2, alpha=2e-05, beta=0.75)
	
	#create the architecture
	def convnn(x, weights, biases, dropout, batchSize):
		layers = []
		a=[]
		for i in range(batchSize):
			a.append(preprocessImg(x[i]))
		x = tf.reshape(a, shape=[-1, 227, 227, 3])
		
		conv0 = conv(x, weights['w0'], biases['b0'], strideSize=4, padding='VALID')
		conv0 = maxpoolnn(conv0)
		conv0 = norm(conv0)

		layers.append(conv0)
		
		conv1 = conv(conv0, weights['w1'], biases['b1'], group=2)
		conv1 = maxpoolnn(conv1)
		conv1 = norm(conv1)
		
		layers.append(conv1)
	
		conv2 = conv(conv1, weights['w2'], biases['b2'])
	
		layers.append(conv2)

		conv3 = conv(conv2, weights['w3'], biases['b3'], group=2)
	
		layers.append(conv3)

		conv4 = conv(conv3, weights['w4'], biases['b4'], group=2)
		conv4 = maxpoolnn(conv4)

		layers.append(conv4)

		fc5 = tf.reshape(conv4, [-1, weights['wf5'].get_shape().as_list()[0]])#shape[0]])
		fc5 = tf.nn.relu_layer(fc5, weights['wf5'], biases['bf5'])
		fc5 = tf.nn.dropout(fc5, dropout)
	
		layers.append(fc5)

		fc6 = tf.nn.relu_layer(fc5, weights['wf6'], biases['bf6'])
		fc6 = tf.nn.dropout(fc6, dropout)

		layers.append(fc6)

#		fc7 = tf.nn.relu_layer(fc6, weights['wf7'], biases['bf7'])
#		fc7 = tf.nn.dropout(fc7, dropout)

#		layers.append(fc7)

		out = tf.add(tf.matmul(fc6, weights['out']), biases['out'])
		out = tf.nn.softmax(out)
		
		layers.append(out)

		return out, layers
	
	pred, allLayers = convnn(x, weights, biases, keepProb, batchSize=bSize)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
	correctPred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


#	answers=np.array([[1,0],[1,0]])#,[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]])
#	init = tf.global_variables_initializer()
#	sess.run(init)
	if train:
		init = tf.global_variables_initializer()
		sess.run(init)
		for i in range(data.shape[0]-5):
			c = sess.run([optimizer, cost, accuracy], feed_dict={x:np.array(data[i:i+5]), y:np.array(answers[i:i+5]), keepProb:dropout})
			print(c)
	else:
		layers = sess.run([allLayers], feed_dict={x:data, keepProb:1.})
		return layers

#loads weights if they're in a single npy file with the biases
def loadWeightsAndBiases():
	path='/home/willarjc/caffe-tensorflow/mydata.npy'
	data = np.load(path)
	data = data.item()
	layerArr=['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']
	weightName=['w0','w1','w2','w3','w4','wf5','wf6','out']
	biasName=['b0', 'b1', 'b2', 'b3', 'b4', 'bf5', 'bf6', 'out']
	weights={}
	biases={}
	for i in range(len(layerArr)):
		weights[weightName[i]] = tf.convert_to_tensor(data[layerArr[i]]['weights'])
	#currently not training the filters
	#	weights[weightName[i]]=tf.Variable(data[layerArr[i]]['weights'], name=weightName[i])
		biases[biasName[i]]=tf.Variable(data[layerArr[i]]['biases'])
	dimension=16
#	weights['wf7'] = tf.get_variable('wf7', shape=[4096, dimension], initializer=tf.contrib.layers.xavier_initializer())
#	biases['bf7'] = tf.Variable(tf.random_normal([dimension]))
#	weights['out'] = tf.get_variable('out',shape=[dimension,2], initializer=tf.contrib.layers.xavier_initializer())
#	biases['out'] = tf.Variable(tf.random_normal([2]))
	return weights, biases

weights, biases = loadWeightsAndBiases()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

bS=450
vBS=1

print("before data loads")
dataObject = dh.COCOData(batchsize=int(bS/2), vbatchsize=vBS)

dataObject.dataLoadByName(objectIDNames[0])
dataObject.loadPureDataByIdName(objectIDNames[0])

pureD, pureL = dataObject.pureData[0], np.asarray([[0,1] for i in range(len(dataObject.pureData[0]))])
anythingElseD, anythingElseL = dataObject.dataLoadAnythingElse()
print("after data loads")

dataPointP = getDataPoint(pureD[0:int(bS/2)], weights, biases, int(bS/2), answers=pureL[0:int(bS/2)])
dataPointAE = getDataPoint(anythingElseD[0:int(bS/2)], weights, biases, int(bS/2), answers=anythingElseL[0:int(bS/2)])

print(dataPointP[0][0].shape)

conv0 = np.zeros(dataPointP[0][0][0].shape)
conv1 = np.zeros(dataPointP[0][1][0].shape)
conv2 = np.zeros(dataPointP[0][2][0].shape)
conv3 = np.zeros(dataPointP[0][3][0].shape)
conv4 = np.zeros(dataPointP[0][4][0].shape)
fc0 = np.zeros(dataPointP[0][5][0].shape)
fc1 = np.zeros(dataPointP[0][6][0].shape)
fc2 = np.zeros(dataPointP[0][7][0].shape)
print(conv0.shape)
print(conv1.shape)
print(conv2.shape)
print(conv3.shape)
print(conv4.shape)
print(fc0.shape)
print(fc1.shape)
print(fc2.shape)
entropyPerNeuron = [conv0, conv1, conv2, conv3, conv4, fc0, fc1, fc2]
entropyPerNeuronPosTracker = [np.empty_like(conv0), np.empty_like(conv1), np.empty_like(conv2), np.empty_like(conv3), np.empty_like(conv4), np.empty_like(fc0),np.empty_like(fc1), np.empty_like(fc2)] 

for count,(layerP,layerAE) in enumerate(zip(dataPointP[0], dataPointAE[0])):
	if count>4:
		for x in range(layerP[0].shape[0]):
			sortedP = np.sort(layerP[:,x])	
			sortedAE = np.sort(layerAE[:,x])
			ToG=np.concatenate((sortedP, sortedAE))
			sortedToG=np.argsort(ToG)[-100:]
			posCount=len(np.where(sortedToG<len(sortedP))[0])
			if posCount==100:
				entropy=0
			else:
				entropy = -((posCount/100)*np.log(posCount/100)+((100-posCount)/100)*np.log((100-posCount)/100))
			entropyPerNeuron[count][x]=entropy
			entropyPerNeuronPosTracker[count][x]=1 if posCount>50 else 0
	else:
		for featureMap in range(layerP[0].shape[2]):
			for x in range(layerP[0].shape[1]):
				for y in range(layerP[0].shape[0]):
					sortedP=np.sort(layerP[:,x,y,featureMap])#[-100:]
					sortedAE=np.sort(layerAE[:,x,y,featureMap])#[-100:]
					ToG = np.concatenate((sortedP,sortedAE))
					sortedToG = np.argsort(ToG)[-100:]
					posCount=len(np.where(sortedToG<len(sortedP))[0])
					if posCount==100:
						entropy=0
					else:
						entropy = -((posCount/100)*np.log(posCount/100)+((100-posCount)/100)*np.log((100-posCount)/100))
					entropyPerNeuron[count][x][y][featureMap]=entropy
					entropyPerNeuronPosTracker[count][x][y][featureMap]= 1 if int(posCount)>50 else 0
for i in range(len(entropyPerNeuron)):
	print("Layer: {}".format(i))
	if i > 4:
		perNeuronInLayer = entropyPerNeuron[i]
		mean = np.mean(perNeuronInLayer)
		stddev = np.std(perNeuronInLayer)
		differentiatingN = np.where(perNeuronInLayer<(mean-stddev))[0]
#		print(differentiatingN)
		for j in range(len(dataPointP[0][i])):
			purePicLayeri = dataPointP[0][i]
			purePicLayeriPicj = dataPointP[0][i][j]
			mean = np.mean(purePicLayeriPicj)
			topFeatures = np.argsort(purePicLayeriPicj)[-len(differentiatingN):]
	else:
		perNeuronInLayer = np.mean(np.mean(entropyPerNeuron[i], axis=1), axis=0)
		mean = np.mean(perNeuronInLayer)
		stddev = np.std(perNeuronInLayer)
		differentiatingN = np.where(perNeuronInLayer<(mean-stddev))[0]
#		print(differentiatingN)
		for entry in range(len(differentiatingN)):
			temp = entropyPerNeuronPosTracker[i][:,:,differentiatingN[entry]]
#			print(round(np.mean(temp),0))
#		print(len(differentiatingN))
#		print(len(perNeuronInLayer))
#		print(len(differentiatingN)/len(perNeuronInLayer))

		for j in range(len(dataPointP[0][i])):
			PurePicLayeri = dataPointP[0][i]
			PurePicLayeriPicj = dataPointP[0][i][j]
			meanPerFeatureMap = np.mean(np.mean(PurePicLayeriPicj, axis=1), axis=0)
			topFeatures = np.argsort(meanPerFeatureMap)[-len(differentiatingN):]
	count=0
	for en in differentiatingN:
		if en in topFeatures:
			count+=1
	if count/len(topFeatures) > .1:
	#	print(topFeatures)
		print(count/len(topFeatures))
	print("~~~~~~~~~")


#print("Accuracy: {:.6f}% at Training Set Size: {}".format(correct/(len(randomValidationD))*100, bS))
#print("~~~~~~~~~~~~~~\n\n")

sess.close()
