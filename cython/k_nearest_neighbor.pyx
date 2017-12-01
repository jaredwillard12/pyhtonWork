import cython
import numpy as np
cimport numpy as np

class KNearestNeighbor():

  def __init__(self):
    pass

  def train(self, X, y):
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=1):
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)
    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    cdef int num_test = X.shape[0]
    cdef int num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
       # dists[i,j] = np.sqrt(np.sum(np.square(self.X_train[j]-X[i]))
        pass        #maybe?
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    cdef int i = 0
    cdef int num_test = X.shape[0]
    cdef int num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      dists[i]=np.sqrt(np.sum(np.square(self.X_train-X[i]), axis=1))
      if i%100==0:
        print(i)
    return dists

  def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, int k=1):
    cdef int num_test = dists.shape[0]
    cdef int i = 0
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      indices = np.argsort(dists[i])[0:k]
      closest_y = self.y_train[indices]
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

    #  for j in range(k):
    #    closestIndex = np.argmax(dists[i])
    #    dists[i][closestIndex]=0
    #    labelOfClosest = self.y_train[closestIndex]
    #    closest_y.append(labelOfClosest)
      
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      import collections
      obj = collections.Counter(closest_y)
      y_pred[i] = obj.most_common(1)[0][0]
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred


def load_CIFAR10(path):
    import cPickle
    import os
    xtrain, ytrain, xtest, ytest = [],[],[],[]
    for file in os.listdir(path):
        filen=file
        filename=path+"/"+filen
        if filen.startswith('data'):
            with open(filename, 'rb') as f:
                dictTemp = cPickle.load(f)
                for en in dictTemp[b'data']:
                    xtrain.append(en)
                for en in dictTemp[b'labels']:
                    ytrain.append(en)
        elif filen.startswith('test'):
            with open(filename, 'rb') as f:
                dictTemp = cPickle.load(f)
                for en in dictTemp[b'data']:
                    xtest.append(en)
                for en in dictTemp[b'labels']:
                    ytest.append(en)
    return np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest)

if __name__=='__main__':
    nn = KNearestNeighbor()
    Xtr, Ytr, Xte, Yte = load_CIFAR10('/home/student/pythonWorkspace/CS231n/assignment1/cs231n/datasets/cifar-10-batches-py')

    idx = np.random.randint(10000, size=100)
    Xte = Xte[idx,:]
    Yte = Yte[idx]

    nn.train(Xtr.reshape(Xtr.shape[0], 32*32*3) ,Ytr)
    Yte_predict = nn.predict(Xte.reshape(Xte.shape[0], 32*32*3), k=7)
    print()
    print('accuracy: %f' % (np.mean(Yte_predict==Yte)))
