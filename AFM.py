# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:37:12 2025

@author: emmalu
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Input
from tensorflow.keras.models import Model
import numpy as np
import util
import layer
import datasetProcess
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score



class AFM_Model(Model):
    def __init__(self, feature_columns, mode):
        """
        初始化 AFM 模型。

        :param feature_columns: 特征列，包含稠密特征列和稀疏特征列。
        :param mode: 模式，可选值为 'avg'、'max' 或 'att'。
        """
        super(AFM_Model, self).__init__()
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # 创建稀疏特征的嵌入层
        self.embed_layer = {
            f"emb_{i}": Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # 创建交互层
        self.interaction_layer = layer.Interaction_layer()
        if self.mode == 'att':
            # 如果模式为 'att'，创建注意力层
            self.attention_layer = layer.Attention_layer()
        # 创建输出层
        self.output_layer = Dense(1)

    def call(self, inputs, **kwargs):
        """
        模型的前向传播方法。

        :param inputs: 输入张量。
        :param kwargs: 其他关键字参数。
        :return: 模型的输出。
        """
        if K.ndim(inputs) != 2:
            raise ValueError(f"Unexpected inputs dimensions {K.ndim(inputs)}, expect to be 2 dimensions")

        # 分离稠密输入和稀疏输入
        dense_inputs, sparse_inputs = inputs[0], inputs[1]
        # 获取稀疏特征的嵌入表示
        embed = [self.embed_layer[f"emb_{i}"](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.convert_to_tensor(embed)
        embed = tf.transpose(embed, [1, 0, 2])  # [None, 26, k]

        # 进行成对交互
        embed = self.interaction_layer(embed)

        if self.mode == 'avg':
            # 平均池化
            x = tf.reduce_mean(embed, axis=1)  # (None, k)
        elif self.mode == 'max':
            # 最大池化
            x = tf.reduce_max(embed, axis=1)  # (None, k)
        else:
            # 使用注意力机制
            x = self.attention_layer(embed)  # (None, k)

        # 通过输出层并应用 sigmoid 激活函数
        output = tf.nn.sigmoid(self.output_layer(x))
        return output