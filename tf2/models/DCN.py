# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:20:50 2025

@author: emmalu
"""

import util
import layer
import datasetProcess

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Embedding

from sklearn.metrics import accuracy_score


class DCN(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation, layer_num, w_reg=1e-4, b_reg=1e-4):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
             for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.deep_layer = layer.Deep_layer(hidden_units, output_dim, activation)
        self.cross_layer = layer.Cross_layer(layer_num, w_reg=w_reg, reg_b=b_reg)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[0], inputs[1]
        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        x = tf.concat([dense_inputs, sparse_embed], axis=1)

        # Crossing layer
        cross_output = self.cross_layer(x)
        # Dense layer
        dnn_output = self.deep_layer(x)

        x = tf.concat([cross_output, dnn_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(x))
        return output
    
    
if __name__ == '__main__': 
    
    k = 8
    w_reg = 1e-4
    b_reg = 1e-4
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    layer_num = 4
    
    [feature_columns,y_train,y_test,
    x_train_categ, x_test_categ,
    x_train_oht,x_test_oht,
    x_train_cross,x_test_cross,
    x_train_conti, x_test_conti] = datasetProcess.loadDataset("adult")


    ##input:[x_train_conti,x_train_categ]                                   
    X_train = (x_train_conti.values,x_train_categ.values)
    X_test = (x_test_conti.values,x_test_categ.values)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    
    
    model = DCN(feature_columns, hidden_units, output_dim, activation=activation, layer_num=layer_num)
    optimizer = optimizers.SGD(0.01)
        
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=2)
    logloss, auc = model.evaluate(X_test, y_test)
    print('logloss {}\nAUC {}'.format(round(logloss,2), round(auc,2)))
    model.summary()