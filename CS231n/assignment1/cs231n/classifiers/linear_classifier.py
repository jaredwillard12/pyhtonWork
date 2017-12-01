from __future__ import print_function

import numpy as np
from sklearn import preprocessing
from linear_svm import *
from softmax import *



class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, Xte, Yte, mean_Image=None, normalize='max', learning_rate=1e-3, reg=1e-5, num_iters=1000,
            batch_size=256, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    if normalize=='max':
        X = preprocessing.normalize(X, norm='max')
    elif normalize=='mean':
        X -= mean_Image
        Xte -= mean_Image
        
    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      print(it, end='\r')
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      indices = np.random.choice(dim, batch_size, replace=True)
      X_batch = X[indices]
      y_batch = y[indices]
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      self.W += -learning_rate*grad
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        y_pred, accuracy = self.predict(Xte, Yte)
        print('iteration %d / %d: loss %f accuracy %.4f' % (it, num_iters, loss, accuracy))

    return loss_history

  def predict(self, X, y):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    for i in range(X.shape[0]):
        allPred = X[i].dot(self.W)
        y_pred[i] = np.argmax(allPred)
    
    accuracy = len(np.where(y_pred==y)[0])/len(y_pred)

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred,accuracy
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
#    lsvm = LinearSVM(self.W)
#    return lsvm.loss(X_batch, y_batch, reg)
    softmax = Softmax(self.W)
    return softmax.loss(X_batch, y_batch, reg)

class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """
  def __init__(self, W):
      self.W = W

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def __init__(self, W):
      self.W = W

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

