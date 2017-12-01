from __future__ import print_function
from softmax import softmax_loss_vectorized as slv

import numpy as np

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size1, hidden_size2, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.grads = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size1)
    self.params['b1'] = np.zeros(hidden_size1)
    self.params['W2'] = std * np.random.randn(hidden_size1, hidden_size2)
    self.params['b2'] = np.zeros(hidden_size2)
    self.params['W3'] = std * np.random.randn(hidden_size2, output_size)
    self.params['b3'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.01, p=0.9, verbose=False):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    
    activF = lambda x: 1.0/(1.0 + np.exp(-x))
    h1 = X.dot(W1) + b1
    ah1 = activF(h1)
    self.ah1 = ah1
#    U1 = (np.random.rand(*ah1.shape) < p) / p
#    ah1 *= U1
    h2 = ah1.dot(W2) +b2
    ah2 = activF(h2)
    self.ah2 = ah2
    score = ah2.dot(W3) + b3
    maxes = np.max(score,axis=1,keepdims=True)
    score -= maxes
    eToTheX = np.exp(score)
    sumD = np.sum(eToTheX, axis=1, keepdims=True)
    scores = eToTheX/sumD

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    
    numTraining = X.shape[0]
    numClasses = scores.shape[1]
    loss = -np.log(scores[np.arange(numTraining),y])
    loss = np.sum(loss)/numTraining + (0.5*reg*np.sum(W1*W1)) + (0.5*reg*np.sum(W2*W2)) + (0.5*reg*np.sum(W3*W3))

    # Backward pass: compute gradients
    grads = {}

    #   X -----
    #           --
    #             --
    #               DOTPRODUCT1(h1)--Sig(h1)*U1--
    #             --                             -- 
    #           --                                 --DOTPRODUCT2(score)---[softmax(score)=scores----HingeLoss]
    #   W1 -----                               ----
    #                                      ----
    #   W2 --------------------------------


    dScores=scores
    dScores[range(numTraining),y]-=1
    dScores/=numTraining

    dDot3dSig=W3.T
    dDot3dW3=(ah2).T
    dSigdh2=((1-ah2)*ah2)
    dScoresdW3=dDot3dW3.dot(dScores)
    dScoresdh2=dScores.dot(dDot3dSig)*dSigdh2

    dDot2dSig=W2.T                              #Derivative of dotProduct2 with respect to the Sigmoid
    
    dSigdh1=((1-ah1)*ah1)                       #Derivative of sigmoid with respect to h1

    dDot2dW2=(ah1).T

    dScoresdW2=dDot2dW2.dot(dScoresdh2)         #Derivative of Scores with respect to W2
    
    dDot1dW1=X.T                                #Derivative of dotProdut1 with respect to W1

    dScoresdh1 = dScoresdh2.dot(dDot2dSig) * dSigdh1            #Derivate of Scores with respect to h1

    dScoresdW1 = dDot1dW1.dot(dScoresdh1)                      #Derivate of Scores with respect to W1
   
    dScoresdW3 += reg * W3
    dScoresdW2 += reg * W2
    dScoresdW1 += reg * W1

    grads['W3']=dScoresdW3
    grads['W2']=dScoresdW2
    grads['W1']=dScoresdW1
    grads['b3']=np.sum(dScores, axis=0, keepdims=True)
    grads['b2']=np.sum(dScoresdh2, axis=0, keepdims=True)
    grads['b1']=np.sum(dScoresdh1, axis=0)

    self.grads['W3']=grads['W3']
    self.grads['W2']=grads['W2']
    self.grads['W1']=grads['W1']
    self.grads['b3']=grads['b3']
    self.grads['b2']=grads['b2']
    self.grads['b1']=grads['b1']


    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=1000,
            batch_size=256, verbose=False, verbose2=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      indices = np.random.choice(X.shape[0], batch_size, replace=True)
      X_batch = X[indices]
      y_batch = y[indices]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, verbose=verbose2)
      loss_history.append(loss)

      self.params['W1'] += -learning_rate*grads['W1']
      self.params['W2'] += -learning_rate*grads['W2']
      self.params['W3'] += -learning_rate*grads['W3']
      self.params['b1'] += -learning_rate*grads['b1']
      self.params['b2'] += -learning_rate*grads['b2'].ravel()
      self.params['b3'] += -learning_rate*grads['b3'].ravel()

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        ans = self.predict(X_batch)
        ans2 = np.argmax(ans[np.arange(X_batch.shape[0])],axis=1)
        train_acc = len(np.where(ans2 == y_batch)[0])/len(y_batch)
        ans = self.predict(X_val)
        ans2 = np.argmax(ans[np.arange(X_val.shape[0])],axis=1)
        val_acc = len(np.where(ans2 == y_val)[0])/len(y_val)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay
        print("\nEPOCH: Validation accuracy: %f\n" % (val_acc))
    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = self.loss(X,p=1)

    return y_pred


