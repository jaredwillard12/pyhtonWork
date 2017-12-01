from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.engine.topology import Layer
import keras.backend as K  
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects


def eednstep(x,name=None):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": "EednStepGrad"}):
        y = tf.greater(x,0.0)
        y = tf.cast(y,x.dtype)
        return tf.identity(x) + tf.stop_gradient(y-x)

@ops.RegisterGradient("EednStepGrad")
def eednsteptestgrad(op, grad):
    x = op.inputs[0]
    #x = tf.Print(x,[x])
    out = tf.maximum(0.0,1.0-tf.abs(x))
    return out*grad # zero out to see the difference:
    
get_custom_objects().update({'eednstep': Activation(eednstep)})


def switch(condition, t, e):
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.where(condition, t, e)
    elif K.backend() == 'theano':
        import theano.tensor as tt
        return tt.switch(condition, t, e)


def _ternarize(W, W_old, hysteresis=0.1, H=1):
    '''The weights' ternarization function, 
    # References:
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902)
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    - Copied from https://github.com/DingKe/nn_playground/tree/master/ternarynet
    '''
    W /= H

    ones = K.ones_like(W)
    zeros = K.zeros_like(W)
    Wt = switch(W >= 0.5+hysteresis, ones, switch(W <= -0.5-hysteresis, -ones,
                                                  switch(tf.logical_or(W >= -0.5+hysteresis,W <= 0.5-hysteresis),
                                                         W_old, zeros)))
    Wt *= H
    

    tf.assign(W_old,Wt)
    return Wt


def ternarize(W, W_old, hysteresis=0.1, H=1):
    '''The weights' ternarization function, 
    # References:
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902)
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    - Copied from https://github.com/DingKe/nn_playground/tree/master/ternarynet
    '''
    Wt = _ternarize(W, W_old, hysteresis, H)
    return W + K.stop_gradient(Wt - W)