# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:08:43 2025

@author: emmalu
"""

import util
import layer
import pandas as pd
import numpy as np

def label_process(train_data,test_data,dt_name = ""):
    if dt_name == 'adult':
        ##根据需求自定义
        train_label = train_data['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
        test_label = test_data['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    return train_label,test_label


def dataProcess(train_data,test_data,LABEL_COLUMN,
                CATEGORICAL_COLUMNS,CROSS_COLUMNS,CONTINUOUS_COLUMNS,embed_dim,dt_name = ""):
    
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
    train_data['label'],test_data['label'] = label_process(train_data,test_data,dt_name)
    y_train = train_data['label']
    y_test = test_data['label']
    
    feature_columns = [[util.denseFeature(feat) for feat in CONTINUOUS_COLUMNS]] + \
               [[util.sparseFeature(feat+"_id", train_data[feat].nunique(), embed_dim) for feat in CATEGORICAL_COLUMNS]]
    
    
    return [feature_columns,y_train,y_test,
            x_train_categ, x_test_categ,
            x_train_oht,x_test_oht,
            x_train_cross,x_test_cross,
            x_train_conti, x_test_conti]



def loadDataset(dt_name):
    if dt_name == 'adult':
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
        data_path = [train_path,test_path]
        embed_dim = 8
        
        train_data,test_data = util.pdReadDataset(data_path,COLUMNS,train_size = 1)
        return dataProcess(train_data,test_data,LABEL_COLUMN,
                        CATEGORICAL_COLUMNS,CROSS_COLUMNS,CONTINUOUS_COLUMNS,embed_dim,dt_name)
                                                   
                            
    
    
# def data_preparation():
#     # Synthetic data parameters
#     num_dimension = 100
#     num_row = 12000
#     c = 0.3
#     rho = 0.8
#     m = 5

#     # Initialize vectors u1, u2, w1, and w2 according to the paper
#     mu1 = np.random.normal(size=num_dimension)
#     mu1 = (mu1 - np.mean(mu1)) / (np.std(mu1) * np.sqrt(num_dimension))
#     mu2 = np.random.normal(size=num_dimension)
#     mu2 -= mu2.dot(mu1) * mu1
#     mu2 /= np.linalg.norm(mu2)
#     w1 = c * mu1
#     w2 = c * (rho * mu1 + np.sqrt(1. - rho ** 2) * mu2)

#     # Feature and label generation
#     alpha = np.random.normal(size=m)
#     beta = np.random.normal(size=m)
#     y0 = []
#     y1 = []
#     X = []

#     for i in range(num_row):
#         x = np.random.normal(size=num_dimension)
#         X.append(x)
#         num1 = w1.dot(x)
#         num2 = w2.dot(x)
#         comp1, comp2 = 0.0, 0.0

#         for j in range(m):
#             comp1 += np.sin(alpha[j] * num1 + beta[j])
#             comp2 += np.sin(alpha[j] * num2 + beta[j])

#         y0.append(num1 + comp1 + np.random.normal(scale=0.1, size=1))
#         y1.append(num2 + comp2 + np.random.normal(scale=0.1, size=1))

#     X = np.array(X)
#     data = pd.DataFrame(
#         data=X,
#         index=range(X.shape[0]),
#         columns=['x{}'.format(it) for it in range(X.shape[1])]
#     )

#     train_data = data.iloc[0:10000]
#     train_label = [y0[0:10000], y1[0:10000]]
#     validation_data = data.iloc[10000:11000]
#     validation_label = [y0[10000:11000], y1[10000:11000]]
#     test_data = data.iloc[11000:]
#     test_label = [y0[11000:], y1[11000:]]

#     return train_data, train_label, validation_data, validation_label, test_data, test_label

# train_data, train_label, validation_data, validation_label, test_data, test_label = data_preparation()
    