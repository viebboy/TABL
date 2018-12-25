#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Dat Tran (dat.tranthanh@tut.fi)
"""
import Models
import keras
import numpy as np


# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40,10], [120,5], [3,1]]

# random data
x = np.random.rand(1000,40,10)
y = keras.utils.to_categorical(np.random.randint(0,3,(1000,)),3)

# get Bilinear model
projection_regularizer = None
projection_constraint = keras.constraints.max_norm(3.0,axis=0)
attention_regularizer = None
attention_constraint = keras.constraints.max_norm(5.0, axis=1)
dropout = 0.1

 
model = Models.TABL(template, dropout, projection_regularizer, projection_constraint,
                    attention_regularizer, attention_constraint)
model.summary()

# create class weight
class_weight = {0 : 1e6/300.0,
                1 : 1e6/400.0,
                2 : 1e6/300.0}


# training
model.fit(x,y, batch_size=256, epochs=100, class_weight=class_weight)

