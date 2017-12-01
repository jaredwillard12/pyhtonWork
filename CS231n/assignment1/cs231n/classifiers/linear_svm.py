import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.T.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: 
        loss += margin
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    margin = scores + (-correct_class_score + 1) # delta = 1
    for j in range(num_classes):
        if y[i]==j:
            numPos = len(np.where(margin>0)[0])
            dW[j]+=numPos*(-1 * X[i])
        else:
            dW[j]+=(margin[j]>0)*X[i]
  for j in range(num_classes):
      dW[j]/=num_train
  dW = dW.T

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dW = dW.T
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  numInBatch = X.shape[0]
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(numInBatch), y]
  margin = (scores.T - correct_class_scores).T + 1
  margin[np.arange(numInBatch),y]=0
  loss = np.sum(margin[np.where(margin>0)])/numInBatch
  loss += reg * np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_classes = W.shape[1]
  for i in range(numInBatch):
      marginRow = margin[i]
      for j in range(num_classes):
        if y[i]==j:
            numPos = len(np.where(marginRow>0)[0])
            dW[j]+=numPos*(-1 * X[i])
        else:
            dW[j]+=(marginRow[j]>0)*X[i]
  for j in range(num_classes):
      dW[j]/=numInBatch
  dW = dW.T

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
