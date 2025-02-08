# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:45:02 2025

@author: emmalu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as LEncoder
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import itertools




####同时考虑训练 & 验证 & 测试集存在
def pdReadDataset(datapath,COLUMNS,train_size = 1):
    ##读取时拆分训练测试,避免后面转换时存在leak的问题
    if len(datapath)==2:
        train_data = pd.read_csv(datapath[0], names=COLUMNS) #打开adult.data训练数据集，列名是COLUMNS
        train_data.dropna(how='any', axis=0) #删除带有空值的行，只要有一个空值，就删除整行
        test_data = pd.read_csv(datapath[1], skiprows=1, names=COLUMNS) #打开adult.test测试数据集
        test_data.dropna(how='any', axis=0) #删除带有空值的行，只要有一个空值，就删除整行
        #all_data = pd.concat([train_data, test_data]) #将训练集和测试集拼接在一起
        #train_size = len(train_data)
        
    elif len(datapath) == 1:
        all_data = pd.read_csv(datapath[0], skiprows=1, names=COLUMNS)
        train_size = np.floor(all_data.shape[0]*train_size)
        train_data = all_data.iloc[:train_size]
        test_data = all_data.iloc[train_size:]

    else:
        print("path lenth should be less than 2!")
    
    return train_data,test_data



def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}  #直接带上了one_hot_dim

def denseFeature(feat):
    return {'feat': feat}


####重写后速度非常慢
class LabelEncoder(LEncoder):
    '''   重写LabelEncoder   '''
    # 重写LabelEncoder，将没有在编码规则里的填充Unknown
    # 且不能直接使用fit_transform 否则会多一个未知编码
    def fit(self, y):
        return super(LabelEncoder, self).fit(list(y) + ['Unknown'])

    def fit_transform(self, y):
        return super(LabelEncoder, self).fit_transform(list(y) + ['Unknown'])

    def transform(self, y):
        new_y = ['Unknown' if x not in set(self.classes_) else x for x in y]
        return super(LabelEncoder, self).transform(new_y)


def apply_transformer(transformer, train_data, test_data):
    train_transformed = transformer.fit_transform(train_data)
    test_transformed = transformer.transform(test_data)
    return train_transformed, test_transformed

def category_encoder(train_data, test_data):
    le = LabelEncoder()
    le.fit(train_data)
    train_data = le.transform(train_data)
    test_data = le.transform(test_data)
    return train_data, test_data


def onehot_encoder(train_data,test_data):
    ohe = OneHotEncoder(handle_unknown='ignore')
    train_data = ohe.fit_transform(train_data)
    test_data = ohe.transform(test_data)
    train_data = pd.DataFrame.sparse.from_spmatrix(train_data,columns = ohe.get_feature_names_out())
    test_data = pd.DataFrame.sparse.from_spmatrix(test_data,columns = ohe.get_feature_names_out())
    return train_data,test_data
    

def polynomial_transform(train_data, test_data):
    ###如果不进行选择，可能造成维度爆炸
    ###但这个函数比较鸡肋, 如果直接使用id类特征,交叉结果具有误导性，如果使用one-hot，同类别内会被进行交叉
    poly = PolynomialFeatures(degree=2)
    train_data, test_data = apply_transformer(poly, train_data, test_data)
    return train_data, test_data

####手动增加一个二阶交叉
def cross_features(onehot_column_names, categories):
    # 按类别分组列名
    column_groups = {}
    for column in onehot_column_names:
        prefix = "_".join(column.split('_')[:-1])
        if prefix in column_groups:
            column_groups[prefix].append(column)
        else:
            column_groups[prefix] = [column]
    # 存储交叉特征的列名
    cross_column_names = []
    cross_column_pairs = []
    for cat1, cat2 in itertools.combinations(categories, 2):
        if cat1 in column_groups and cat2 in column_groups:
            for col1 in column_groups[cat1]:
                for col2 in column_groups[cat2]:
                    cross_column_names.append(f"{col1}*{col2}")
                    cross_column_pairs.append((col1, col2))
    return cross_column_names, cross_column_pairs


def compute_cross_features(df, cross_column_pairs):
    cross_features = {}
    for col1, col2 in cross_column_pairs:
        cross_features[f"{col1}*{col2}"] = df[col1] * df[col2]
    cross_df = pd.DataFrame(cross_features)
    return cross_df

def standard_scaler(train_data, test_data):
    ##全量标准化
    ##其他标准化的方法后续再进行补充
    scaler = StandardScaler()
    train_data, test_data = apply_transformer(scaler, train_data, test_data)
    return train_data, test_data


def calculate_missing_values(train_data, test_data, c):
    test_value_counts = test_data[c].value_counts().reset_index() ##不考虑na
    test_value_counts['ratio'] = test_value_counts['count']/test_value_counts['count'].sum()
    train_unique = pd.DataFrame(train_data[c].unique(), columns=[c])
    merged = pd.merge(test_value_counts, train_unique, on=c, how='left', indicator=True)
    merged = merged[merged['_merge'] == 'left_only']
    if merged.shape[0] == 0:
        print("列%s 没有新增类别"%(c))
    else:
        print("测试集存在新增类别！！！")
        print(merged)
