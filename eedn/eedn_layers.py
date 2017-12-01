import numpy as np

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from keras import backend as K
from keras import constraints
from keras import initializers
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Flatten, Activation, ActivityRegularization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import activations
from keras import initializers
from eedn_ops import ternarize, switch

class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}

    
class PredictionInit(initializers.Initializer):
    """Initializer that assigns each filter to a class and returns matrix that 
       weights the average inputs.
    """
    def __init__(self, classes,inshape):
        self.classes = classes
        self.inshape = inshape
        
    def __call__(self, shape, dtype=None):
        filters = self.inshape[-1]
        classes = self.classes
        a = np.zeros(self.inshape)
        for i in range(filters):
            a[:,:,i] = i%classes
            
        flat = a.reshape([-1])
        
        y = np.zeros((flat.shape[0],classes))
        assert (y.shape == shape)
        
        for i in range(classes):
            idxs = np.where(flat==i)[0]
            y[idxs,i] = 1.0/(idxs.shape[0])
            
        return y

    def get_config(self):
        return {
            'classes': self.classes,
            'inshape': self.inshape
        }

class RandomTernary(initializers.Initializer):
    """Initializer that generates ternay tensors [minval,0,maxval] from a uniform distribution.
    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, minval=-1, maxval=1, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None):
        vals = K.random_uniform(shape, self.minval, self.maxval,
                                dtype=dtype, seed=self.seed)
        ones = K.ones_like(vals)
        zeros = K.zeros_like(vals)

        return switch(vals >= 0.5, ones, switch(vals <= -0.5, -ones, zeros))
    
    def get_config(self):
        return {
            'minval': self.minval,
            'maxval': self.maxval,
            'seed': self.seed,
        }

class EednPrediction(Layer):
    """The last Eedn neuron layer typically contains multiple neurons that are assigned to each of the available class labels.
       The role of the prediction layer is to compute the average from all such neurons and output a single activation level
       for each of the classes. For that reason, a Eedn network must have at most a single prediction layer and it must follow
        the last neuron layer in the network.
    # Arguments
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """
    
    def __init__(self, units, activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(EednPrediction, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.input_spec = InputSpec(min_ndim=4)
        self.supports_masking = True

    def build(self, input_shape):
        print(input_shape)
        assert len(input_shape) == 4
        input_dim = np.prod(input_shape[1:])
     
        self.kernel_initializer = PredictionInit(self.units, input_shape[1:])
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=False,
                                      name='kernel')
        
        self.input_spec = InputSpec(min_ndim=4, max_ndim=4)
        self.built = True

    def call(self, inputs):
        output = K.batch_flatten(inputs)
        output = K.dot(output, self.kernel)
     
        if self.activation is not None:
            output = self.activation(output)
          
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
        }
        base_config = super(EednPrediction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class Bias(Layer):
    """Just adds biases to a NN layer.
    `Bias` implements the operation:
    `output = activation(input + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, and `bias` is a bias vector created by the layer.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,), activation=None, use_bias=False)
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Bias(activation='relu'))
        model.add(Dense(32))
    ```
    # Arguments
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """
    
    def __init__(self, activation=None, bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Bias, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.bias = self.add_weight(shape=(input_dim,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.bias_add(inputs,self.bias)
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class EednConv2D(Conv2D):
    '''Eedn Convolution2D layer
    References: 
    - [Convolutional Networks for Fast, Energy-Efficient Neuromorphic Computing](https://arxiv.org/abs/1603.08270)}
    - Modifed from https://github.com/DingKe/nn_playground/tree/master/ternarynet
    '''
    def __init__(self, filters, groups = 1, hysteresis=0.1, H=1.0, bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None, useMix=False, **kwargs):
        super(EednConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.hysteresis = hysteresis
        self.groups = groups
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.useMix = useMix
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
            
        if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        
        if input_dim % self.groups != 0:
            raise ValueError('The channels must evenly divide the groups')

        input_per_group = input_dim // self.groups
        
        kernel_shape = self.kernel_size + (input_per_group, self.filters)
        
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        
        self.kernel_old = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel_old',
                                 constraint=self.kernel_constraint,
                                 trainable=False)
        
        if self.useMix:
            self.mix = np.random.permutation(input_dim).astype(np.int32)
            
        if self.use_bias:
            self.bias = self.add_weight((self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.data_format == 'channels_first':
            channel_axis = 0
        else:
            channel_axis = -1 
            
        ternary_kernel = ternarize(self.kernel, self.kernel_old, hysteresis = self.hysteresis, H=self.H)
        
        if self.groups == 1:
            outputs = K.conv2d(
                        inputs,
                        self.kernel,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate)
        else:
            
            if self.useMix:
                inputs_shuffle = tf.gather(inputs,self.mix,axis=channel_axis)
            else:
                inputs_shuffle = inputs
            input_slices = tf.split(inputs_shuffle, self.groups, axis=channel_axis)
            kernel_slices = tf.split(ternary_kernel, self.groups, axis=channel_axis)
            output_slices = [K.conv2d(
                                input_slice,
                                kernel_slice,
                                strides=self.strides,
                                padding=self.padding,
                                data_format=self.data_format,
                                dilation_rate=self.dilation_rate)
                             for input_slice, kernel_slice in zip(input_slices, kernel_slices)]

            outputs = tf.concat(output_slices, axis=channel_axis)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        
        return outputs
        
    def get_config(self):
        config = {'H': self.H,
                  'groups': self.groups,
                  'hysteresis': self.hysteresis}
        base_config = super(EednConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def eednLayer(model, filters, kernel_size, strides, groups=1, input_shape=None, transduction=False, padding = 'valid', H=1, keras_conv = False):
    if transduction:
        kernel_constraint=None
        kernel_initializer=initializers.RandomUniform(-1,1)
    else:
        kernel_constraint=Clip(-H, H)
        kernel_initializer=RandomTernary()

    if keras_conv:
        if input_shape == None:
            model.add(Conv2D(filters, kernel_size = kernel_size, strides=strides,
                      padding=padding, use_bias=False,
                      kernel_constraint=kernel_constraint,
                      kernel_initializer=kernel_initializer))
        else:   
            model.add(Conv2D(filters, kernel_size = kernel_size, input_shape = input_shape,
                         padding=padding, use_bias=False, strides=strides,
                         kernel_constraint=kernel_constraint,
                         kernel_initializer=kernel_initializer))
    else:
        if input_shape == None:
            model.add(EednConv2D(filters, kernel_size = kernel_size, strides=strides,
                      groups=groups, padding=padding, use_bias=False,
                      kernel_constraint=kernel_constraint,
                      kernel_initializer=kernel_initializer))                            
        else:   
            model.add(EednConv2D(filters, kernel_size = kernel_size, input_shape = input_shape,
                      groups=groups, padding=padding, use_bias=False, strides=strides,
                      kernel_constraint=kernel_constraint,
                      kernel_initializer=kernel_initializer))

    model.add(BatchNormalization())
    model.add(Activation('eednstep'))
 #   model.add(ActivityRegularization(l2 = 0.00001))
 