import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  dW = dW.T
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  numBatches = X.shape[0]
  numClasses = W.shape[1]
  for i in range(numBatches):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    eToTheX = np.exp(scores)
    sumD = np.sum(eToTheX)
    answer = eToTheX/sumD
    loss += -answer[y[i]]+np.log(sumD)
    for j in range(len(answer)):
        p = answer[j]
        dW[j] += (p-(j==y[i]))*X[i]
  loss /= numBatches
  dW /= numBatches
  dW=dW.T

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW = dW.T
  numTraining = X.shape[0]
  numClasses = W.shape[1]
  scores = X.dot(W)
  maxes = np.max(scores, axis=1)
  scores = scores.T-maxes
  scores = scores.T
  eToTheX = np.exp(scores)
  sumD = np.sum(eToTheX, axis=1)
  answer = eToTheX.T/sumD
  answer = answer.T
  loss = np.sum(-answer[np.arange(len(answer)),y]+np.log(sumD[np.arange(len(answer))]))
#  I = np.identity(numTraining)
#  dL = np.zeros(answer.shape)
#  dL = -answer*(I[:,np.arange(numClasses)]-answer)
  for i in range(numTraining):
    answerTemp=answer[i]
    for j in range(numClasses):
      p = answerTemp[j]
      dW[j] += (p-(j==y[i]))*X[i]

#  dW=X.T.dot(dL)

  loss /= numTraining
  dW /= numTraining
  dW=dW.T

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

