#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Dat Tran (dat.tranthanh@tut.fi)
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations as Activations
from keras import initializers as Initializers


class Constraint(object):
    """
    Constraint template
    """
    def __call__(self, w):
        return w

    def get_config(self):
        return {}
    
class MinMax(Constraint):
    """
    Customized min-max constraint for scalar
    """

    def __init__(self, min_value=0.0, max_value=10.0):
        self.min_value = min_value
        self.max_value = max_value
    def __call__(self, w):
        
        return K.clip(w,self.min_value,self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}
        

def nmodeproduct(x,w,mode):
    """
    n-mode product for 2D matrices
    x: NxHxW
    mode=1 -> w: Hxh
    mode=2 -> w: Wxw
    
    output: NxhxW (mode1) or NxHxw (mode2)
    """
    if mode==2:
        x=K.dot(x,w)
    else:
        x=K.permute_dimensions(x,(0,2,1))
        x=K.dot(x,w)
        x=K.permute_dimensions(x,(0,2,1))
    return x

class BL(Layer):
    """
    Bilinear Layer
    """
    def __init__(self, output_dim,
                 kernel_regularizer=None,
                 kernel_constraint=None,**kwargs):
        """
        output_dim : output dimensions of 2D tensor, should be a list of len 2, e.g. [30,20]
        kernel_regularizer : keras regularizer object
        kernel_constraint: keras constraint object
        """
        
        self.output_dim = output_dim
        self.kernel_regularizer=kernel_regularizer
        self.kernel_constraint=kernel_constraint
        
        super(BL, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',shape=(input_shape[1], self.output_dim[0]),
                                      initializer='he_uniform',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        self.W2 = self.add_weight(name='W2',shape=(input_shape[2], self.output_dim[1]),
                                      initializer='he_uniform',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        self.bias=self.add_weight(name='bias',shape=(self.output_dim[0],self.output_dim[1]),
                              initializer='zeros',trainable=True)
        super(BL, self).build(input_shape)

    def call(self, x):
        print(K.int_shape(x))
        x = nmodeproduct(x,self.W1,1)
        x = nmodeproduct(x,self.W2,2)
        x = K.bias_add(x,self.bias)
        
        if self.output_dim[1]==1:
            x = K.squeeze(x,axis=-1)
        print(K.int_shape(x))
        return x

    def compute_output_shape(self, input_shape):
        if self.output_dim[1]==1:
            return (input_shape[0], self.output_dim[0])
        else:
            return (input_shape[0], self.output_dim[0], self.output_dim[1])
        
        
class TABL(Layer):
    """
    Temporal Attention augmented Bilinear Layer
    https://arxiv.org/abs/1712.00975
    
    """
    def __init__(self, output_dim,
                 projection_regularizer=None,
                 projection_constraint=None,
                 attention_regularizer=None,
                 attention_constraint=None,
                 **kwargs):
        """
        output_dim : output dimensions of 2D tensor, should be a list of len 2, e.g. [30,20]
        projection_regularizer : keras regularizer object for projection matrix
        projection_constraint: keras constraint object for projection matrix
        attention_regularizer: keras regularizer object for attention matrix
        attention_constraint: keras constraint object for attention matrix
        """
        
        self.output_dim = output_dim
        self.projection_regularizer = projection_regularizer
        self.projection_constraint = projection_constraint
        self.attention_regularizer = attention_regularizer
        self.attention_constraint = attention_constraint
        
        super(TABL, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',shape=(input_shape[1], self.output_dim[0]),
                                      initializer='he_uniform',
                                      regularizer=self.projection_regularizer,
                                      constraint=self.projection_constraint,
                                      trainable=True)
        
        self.W2 = self.add_weight(name='W2',shape=(input_shape[2], self.output_dim[1]),
                                      initializer='he_uniform',
                                      regularizer=self.projection_regularizer,
                                      constraint=self.projection_constraint,
                                      trainable=True)
        
        self.W = self.add_weight(name='W',shape=(input_shape[2], input_shape[2]),
                                      initializer=Initializers.Constant(1.0/input_shape[2]),
                                      regularizer=self.attention_regularizer,
                                      constraint=self.attention_constraint,
                                      trainable=True)
        
        self.alpha = self.add_weight(name='alpha',shape=(1,),
                                      initializer=Initializers.Constant(0.5),
                                      constraint=MinMax(),
                                      trainable=True)


        self.bias=self.add_weight(name='bias',shape=(1, self.output_dim[0],self.output_dim[1]),
                              initializer='zeros',trainable=True)
        
        self.in_shape = input_shape
        
        super(TABL, self).build(input_shape)

    def call(self, x):
        """
        x: Nx D1 x D2
        W1 : D1 x d1
        W2: D2 x d2
        W: D2 x D2
        """
        # first mode projection
        x = nmodeproduct(x,self.W1,1) # N x d1 x D2
        # enforcing constant (1) on the diagonal 
        W = self.W-self.W*K.eye(self.in_shape[2],dtype='float32')+K.eye(self.in_shape[2],dtype='float32')/self.in_shape[2]
        # calculate attention 
        attention = Activations.softmax(nmodeproduct(x,W,2),axis=-1) # N x d1 x D2
        # apply attention
        x = self.alpha*x + (1.0 - self.alpha)*x*attention
        # second mode projection
        x = nmodeproduct(x,self.W2,2)
        # bias add
        x = x + self.bias
        
        if self.output_dim[1]==1:
            x = K.squeeze(x,axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        if self.output_dim[1]==1:
            return (input_shape[0], self.output_dim[0])
        else:
            return (input_shape[0], self.output_dim[0], self.output_dim[1])



