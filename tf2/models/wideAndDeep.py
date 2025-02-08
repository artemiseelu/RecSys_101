# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:53:02 2025

@author: emmalu
"""

import util
import layer

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding

from sklearn.metrics import accuracy_score



class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embedding_layer = {'embed_layer'+str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                for i,feat in enumerate(self.sparse_feature_columns)}

        self.wide = layer.Wide_layer()
        self.deep = layer.Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        # dense_inputs: 数值特征
        # sparse_inputs： 类别特征
        # onehot_inputs：类别特征onehot+指定类别交叉
        dense_inputs, sparse_inputs, onehot_inputs = inputs[0], inputs[1], inputs[2] 

        # wide部分
        wide_input = tf.concat([dense_inputs, onehot_inputs], axis=-1)
        wide_output = self.wide(wide_input)

        # deep部分
        sparse_embed = tf.concat([self.embedding_layer['embed_layer'+str(i)](sparse_inputs[:, i]) \
                        for i in range(sparse_inputs.shape[-1])], axis=-1)
        deep_input = tf.concat([dense_inputs, sparse_embed], axis=-1)
        deep_output = self.deep(deep_input)

        output = tf.nn.sigmoid(0.5*(wide_output + deep_output))
        return output



def label_process(train_data,test_data):
    ##根据需求自定义
    train_label = train_data['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    test_label = test_data['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    return train_label,test_label


def dataProcess(train_data,test_data,LABEL_COLUMN,
                CATEGORICAL_COLUMNS,CROSS_COLUMNS,CONTINUOUS_COLUMNS,embed_dim):
    
    ##填充缺失值
    train_data[CONTINUOUS_COLUMNS] = train_data[CONTINUOUS_COLUMNS].fillna(0)
    train_data[CATEGORICAL_COLUMNS] = train_data[CATEGORICAL_COLUMNS].fillna('NK')
    test_data[CONTINUOUS_COLUMNS] = test_data[CONTINUOUS_COLUMNS].fillna(0)
    test_data[CATEGORICAL_COLUMNS] = test_data[CATEGORICAL_COLUMNS].fillna('NK')
    
    #类别特征id化
    CATEGORICAL_COLUMNS_ID = [i+"_id" for i in CATEGORICAL_COLUMNS]
    for c in CATEGORICAL_COLUMNS:
        util.calculate_missing_values(train_data, test_data, c)
        train_data[c+"_id"],test_data[c+"_id"] = util.category_encoder(train_data[c],test_data[c])
    x_train_categ = train_data[CATEGORICAL_COLUMNS_ID]
    x_test_categ = test_data[CATEGORICAL_COLUMNS_ID]
    
    #类别特征向量化(不能确定有多长）
    x_train_oht,x_test_oht = util.onehot_encoder(train_data[CATEGORICAL_COLUMNS_ID],test_data[CATEGORICAL_COLUMNS_ID])
    ONEHOT_COLUMNS_ID = x_train_oht.columns
    
    #类别特征交叉
    selected_feature_names =[name for name in ONEHOT_COLUMNS_ID if any(keyword in name for keyword in CROSS_COLUMNS)]
    cross_column_names, cross_column_pairs = util.cross_features(selected_feature_names, CATEGORICAL_COLUMNS_ID)
    x_train_cross = util.compute_cross_features(x_train_oht[selected_feature_names], cross_column_pairs)
    x_test_cross = util.compute_cross_features(x_test_oht[selected_feature_names], cross_column_pairs)
    x_train_cross = pd.concat([x_train_oht,x_train_cross],axis = 1)
    x_test_cross = pd.concat([x_test_oht,x_test_cross],axis = 1)
    
    #连续特征标准化
    CONTINUOUS_COLUMNS_STD = [i+"_std" for i in CONTINUOUS_COLUMNS]
    train_data[CONTINUOUS_COLUMNS_STD],test_data[CONTINUOUS_COLUMNS_STD] = util.standard_scaler(train_data[CONTINUOUS_COLUMNS],\
                                                                                           test_data[CONTINUOUS_COLUMNS])
    x_train_conti = train_data[CONTINUOUS_COLUMNS_STD]
    x_test_conti = test_data[CONTINUOUS_COLUMNS_STD]
    
    #获得label
    train_data['label'],test_data['label'] = label_process(train_data,test_data)
    y_train = train_data['label']
    y_test = test_data['label']
    
    feature_columns = [[util.denseFeature(feat) for feat in CONTINUOUS_COLUMNS]] + \
               [[util.sparseFeature(feat+"_id", train_data[feat].nunique(), embed_dim) for feat in CATEGORICAL_COLUMNS]]
    
    
    return [feature_columns,y_train,y_test,
            x_train_categ, x_test_categ,
            x_train_oht,x_test_oht,
            x_train_cross,x_test_cross,
            x_train_conti, x_test_conti]



if __name__ == '__main__':
    train_path = 'E:\\workspace\\RS_project\\dataset\\adult\\adult.data'
    test_path = 'E:\\workspace\\RS_project\\dataset\\adult\\adult.test'
    COLUMNS = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
        "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", 
        "hours_per_week", "native_country", "income_bracket"
    ]
    
    LABEL_COLUMN = "label"
    
    CATEGORICAL_COLUMNS = [
        "workclass", "education", "marital_status", "occupation", "relationship", 
        "race", "gender", "native_country"
    ] #非数字的列，分类数据
    
    CROSS_COLUMNS = ["workclass","gender"]
    
    CONTINUOUS_COLUMNS = [
        "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
    ] #连续数据对应的列
    
    embed_dim = 8
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    
    
    
    data_path = [train_path,test_path]
    train_data,test_data = util.pdReadDataset(data_path,COLUMNS,train_size = 1)
    [feature_columns,y_train,y_test,
    x_train_categ, x_test_categ,
    x_train_oht,x_test_oht,
    x_train_cross,x_test_cross,
    x_train_conti, x_test_conti] = dataProcess(train_data,test_data,LABEL_COLUMN,
                    CATEGORICAL_COLUMNS,CROSS_COLUMNS,CONTINUOUS_COLUMNS,embed_dim)
     
    #W&D输入数据：[dense_f,sparse_f,oht_corss_f]                                           
    X_train = (x_train_conti.values,x_train_categ.values,x_train_cross.values)
    X_test = (x_test_conti.values,x_test_categ.values,x_test_cross.values)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)          
                                               

    model = WideDeep(feature_columns, hidden_units, output_dim, activation)
    #optimizer = optimizers.SGD(0.01)
    
    

    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=2)
    logloss, auc = model.evaluate(X_test, y_test)
    print('logloss {}\nAUC {}'.format(round(logloss,2), round(auc,2)))
    model.summary()
    
    #predictions = model.predict(X_test)
    ## 将概率转换为类别
    #threshold = 0.5
    #predictions = (probabilities > threshold).astype(int)
    #print("预测类别结果：")
    #print(predictions)
    
    # 将概率转换为类别
    #predictions = np.argmax(probabilities, axis=1)
    #print("预测类别结果：")
    #print(predictions)
