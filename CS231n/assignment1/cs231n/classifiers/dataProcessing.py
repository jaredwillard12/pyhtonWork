import numpy as np

class DataProcesser():

    def __init__(self, description, path, directory=True):
        self.directory=directory
        self.description=description
        self.path=path
        self.imgArrays=[]
        self.meanSubtracted=False
        self.normalized=False
        self.PCAed=False
        self.whitened=False

    def loadPickledFilesInDirectory(self):
        if not self.directory:
            return
        import pickle
        import os
        path=self.path
        xtrain, ytrain, xtest, ytest = [],[],[],[]
        for file in os.listdir(path):
            filen=os.fsdecode(file)
            filename=path+"/"+filen
            if filen.startswith('data') or filen.startswith('train'):
                with open(filename, 'rb') as f:
                    dictTemp=pickle.load(f,encoding='bytes')
                    for en in dictTemp[b'data']:
                        xtrain.append(en)
                    for en in dictTemp[b'labels']:
                        ytrain.append(en)
            elif filen.startswith('test') or filen.startswith('val'):
                with open(filename, 'rb') as f:
                    dictTemp=pickle.load(f,encoding='bytes')
                    for en in dictTemp[b'data']:
                        xtest.append(en)
                    for en in dictTemp[b'labels']:
                        ytest.append(en)
        self.imgArrays = [np.array(xtrain, dtype=float), np.array(ytrain, dtype=int), np.array(xtest, dtype=float), np.array(ytest, dtype=int)]

    def meanSubtraction(self):
        train, test = self.imgArrays[0], self.imgArrays[2]
        mean=np.mean(train,axis=0)
        self.imgArrays[0]=np.subtract(train,mean)
        self.imgArrays[2]=np.subtract(test,mean)
        self.meanSubtracted=True

    def normalizeData(self):
        train, test = self.imgArrays[0], self.imgArrays[2]
        std=np.std(train, axis=0)
        self.imgArrays[0]=train/mean
        self.imgArrays[2]=test/mean
        self.normalized=True

    def PCA(self):
        train, test = self.imgArrays[0], self.imgArrays[2]
        if not self.meanSubtracted:
            self.meanSubtraction()
        cov = np.dot(train.T, train) / train.shape[0]
        U,S,V = np.linalg.svd(cov)
        Trainrot = np.dot(train, U) # decorrelate the data
        Trainrot_reduced = np.dot(train, U[:,:100]) # Xrot_reduced becomes [N x 100]
        Testrot = np.dot(test, U) # decorrelate the data
        Testrot_reduced = np.dot(test, U[:,:100]) # Xrot_reduced becomes [N x 100]
        self.imgArrays[0] = Trainrot_reduced
        self.imgArrays[2] = Testrot_reduced
        self.PCAed = True

    def whitenData(self):
        train, test = self.imgArrays[0], self.imgArrays[2]
        if not self.meanSubtracted:
            self.meanSubtraction()
        cov = np.dot(train.T, train) / train.shape[0]
        U,S,V = np.linalg.svd(cov)
        Trainrot = np.dot(train, U) # decorrelate the data
        Trainwhite = Trainrot / np.sqrt(S + 1e-5)
        Testrot = np.dot(test, U) # decorrelate the data
        Testwhite = Testrot / np.sqrt(S + 1e-5)
        self.imgArrays[0]=Testwhite
        self.imgArrays[2]=Testwhite
        self.whitened=True

    if __name__=='__main__':
        init()
