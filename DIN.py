# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:00:44 2025

@author: emmalu
"""

import random
import pickle
import pandas as pd
import numpy as np

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df

def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  #df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key



random.seed(1234)



reviews_df = to_df('E:\\workspace\\RS_project\\dataset\\meta_Electronics\\Electronics_5.json')
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
#with open('E:\\workspace\\RS_project\\dataset\\Electronics_5.pkl', 'wb') as f:
#  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('E:\\workspace\\RS_project\\dataset\\meta_Electronics\\meta_Electronics.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
meta_df = meta_df[['asin', 'categories']]
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1]) ###可能有多类别
#with open('E:\\workspace\\RS_project\\dataset\\meta_Electronics\\meta_Electronics.pkl', 'wb') as f:
#  pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

asin_map, asin_key = build_map(reviews_df, 'asin')
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')


user_count, item_count, cate_count, example_count =\
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

##速度好像非常慢
# reviews_df['asin'],meta_df['asin'] = util.category_encoder(reviews_df['asin'],meta_df['asin'])
# reviews_df['reviewerID'],_ = util.category_encoder(reviews_df['reviewerID'],reviews_df['reviewerID'])
# meta_df['reviewerID'],_ = util.category_encoder(meta_df['categories'],meta_df['categories'])


 
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
meta_df['asin'] = meta_df['asin'].map(lambda x: asin_map[x])
reviews_df['reviewerID'] = reviews_df['reviewerID'].map(lambda x: revi_map[x])
meta_df['categories'] = meta_df['categories'].map(lambda x: cate_map[x])

meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)

# with open('data/amazon/remap.pkl', 'wb') as f:
#   pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
#   pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
#   pickle.dump((user_count, item_count, cate_count, example_count),
#               f, pickle.HIGHEST_PROTOCOL)
#   pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)



# with open('data/amazon/remap.pkl', 'rb') as f:
#   reviews_df = pickle.load(f)
#   cate_list = pickle.load(f)
#   user_count, item_count, cate_count, example_count = pickle.load(f)


train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  def gen_neg():
    neg = pos_list[0]
    # generate item that user doesn't
    while neg in pos_list:
      neg = random.randint(0, item_count-1) #随机选一个负样本
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))] # generate same length negative item sample: balance 1:1

  # 1 reviewer has N positive item: generate N negative item
  # 根据时序生成样本，正负比例1:1
  for i in range(1, len(pos_list)):
    hist = pos_list[:i]
    hist_cate = [cate_list[i] for i in hist]
    if i != (len(pos_list)-1):
      train_set.append((reviewerID, hist, hist_cate, i, pos_list[i], cate_list[pos_list[i]], 1)) # use N-1 for train and 1 for test
      train_set.append((reviewerID, hist, hist_cate, i, neg_list[i], cate_list[neg_list[i]], 0)) # generate N-1 negative sample for train
    else:
      test_set.append((reviewerID, hist, hist_cate, i, pos_list[i], cate_list[pos_list[i]], 1)) # use N-1 for train and 1 for test
      test_set.append((reviewerID, hist, hist_cate, i, neg_list[i], cate_list[neg_list[i]], 0)) # generate N-1 negative sample for train

random.shuffle(train_set)
random.shuffle(test_set)

colunms = ['reviewer_id','hist_item_list','hist_category_list','hist_length','item','item_category','label']
train_df = pd.DataFrame(train_set,columns = colunms)
test_df = pd.DataFrame(test_set,columns = colunms)


# with open('dataset.pkl', 'wb') as f:
#     pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL) #2608764
#     pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL ) #384806
    
# with open( 'dataset.pkl', 'rb' ) as f:
#     train = pickle.load( f )
#     valid = pickle.load( f )


from const import *
import tensorflow as tf


def build_features():
    # everything but the feature used in attention
    f_reviewer = tf.feature_column.categorical_column_with_identity(
        'reviewer_id',
        num_buckets = AMAZON_USER_COUNT,
        default_value = 0
    )
    f_reviewer = tf.feature_column.embedding_column(f_reviewer, dimension = 128)

    f_item_length = tf.feature_column.numeric_column('hist_length')

    f_dense = [f_item_length, f_reviewer]

    return f_dense


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense,BatchNormalization
from tensorflow.keras import backend as K



class AttentionLayer(Layer):
    def __init__(self, hidden_units, activation='relu'):
        super(AttentionLayer, self).__init__()
        self.dense_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(1, activation=None)


    def call(self, inputs):
        queries, keys, values = inputs
        # 与原函数逻辑一致的操作
        padded_size = tf.shape(keys)[1]
        queries = tf.expand_dims(queries, axis=1) 
        queries = tf.tile(queries, [1, padded_size, 1])
        dense = tf.concat([queries, keys, queries - keys, queries * keys], axis=2)

        for dense_layer in self.dense_layer:
            dense = dense_layer(dense)

        weight = self.out_layer(dense) 
        #weight = tf.squeeze(weight, axis=-1)

        zero_mask = tf.expand_dims(tf.not_equal(values, 0), axis=2)
        #zero_mask = tf.squeeze(zero_mask, axis=-1)
        zero_weight = tf.ones_like(weight) * (-2 ** 32 + 1)
        #zero_weight = tf.expand_dims(zero_weight, axis=-1)
        print(zero_mask.shape,weight.shape,zero_weight.shape)
        weight = tf.where(zero_mask, weight, zero_weight)

        weight = tf.nn.softmax(weight, axis=1)
        output = tf.reduce_mean(tf.multiply(weight, keys), axis=1)
        print(output.shape)
        #output = tf.squeeze(output, axis=1) # [None, k]
        return output


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn_layer = BatchNormalization()
        self.alpha = self.add_weight(name='alpha', shape=(1,), trainable=True)

    def call(self, inputs, **kwargs):
        x = self.bn_layer(inputs)
        x = tf.nn.sigmoid(x)
        output = x * inputs + (1-x) * self.alpha * inputs
        return output
    
    
# 示例输入形状
batch_size = 32
emb_dim = 128
padded_size = 10

# 随机生成示例数据
queries = tf.random.normal((batch_size, emb_dim))
keys = tf.random.normal((batch_size, padded_size, emb_dim))
keys_id = tf.random.uniform((batch_size, padded_size), minval=0, maxval=2, dtype=tf.int32)

# 定义注意力隐藏单元
attention_hidden_units = [64, 32]

# 创建自定义注意力层
attention_layer = AttentionLayer(attention_hidden_units)
# 调用层
output = attention_layer([queries, keys, keys_id])

print("Output shape:", output.shape)





class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units, ffn_hidden_units,
                 att_activation='prelu', ffn_activation='prelu', maxlen=40, dnn_dropout=0.0):
        super(DIN, self).__init__()
        self.maxlen = maxlen
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        self.other_sparse_num = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_num = len(self.dense_feature_columns)
        self.behavior_num = len(behavior_feature_list)

        # other sparse embedding
        self.embed_sparse_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                    for feat in self.sparse_feature_columns
                                      if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                    for feat in self.sparse_feature_columns
                                      if feat['feat'] in behavior_feature_list]

        self.att_layer = Attention_Layer(att_hidden_units, att_activation)
        self.bn_layer = BatchNormalization(trainable=True)
        self.dense_layer = [Dense(unit, activation=PReLU() if ffn_activation=='prelu' else Dice())\
             for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None):
        # dense_inputs:  empty/(None, dense_num)
        # sparse_inputs: empty/(None, other_sparse_num)
        # history_seq:  (None, n, k)
        # candidate_item: (None, k)
        dense_inputs, sparse_inputs, history_seq, candidate_item = inputs[0],

        # dense & sparse inputs embedding
        other_feat = tf.concat([layer(sparse_inputs[:, i]) for i, layer in enumerate(self.embed_sparse_layers)],
                            axis=-1)
        other_feat = tf.concat([other_feat, dense_inputs], axis=-1)

        # history_seq & candidate_item embedding
        seq_embed = tf.concat([layer(history_seq[:, :, i])
                            for i, layer in enumerate(self.embed_seq_layers)],
                            axis=-1)   # (None, n, k)
        item_embed = tf.concat([layer(candidate_item[:, i])
                            for i, layer in enumerate(self.embed_seq_layers)],
                            axis=-1)   # (None, k)

        # one_hot之后第一维是1的token，为填充的0
        mask = tf.cast(tf.not_equal(history_seq[:, :, 0], 0), dtype=tf.float32)   # (None, n)
        att_emb = self.attention_layer([item_embed, seq_embed, seq_embed, mask])  # (None, k)

        # 若其他特征不为empty
        if self.dense_len>0 or self.other_sparse_len>0:
            emb = tf.concat([att_emb, item_embed, other_feat], axis=-1)
        else:
            emb = tf.concat([att_emb, item_embed], axis=-1)

        emb = self.bn_layer(emb)
        for layer in self.dense_layer:
            emb = layer(emb)

        emb = self.dropout(emb)
        output = self.out_layer(emb)
        return tf.nn.sigmoid(output) # (None, 1)
    
    
#https://github.com/littleqiezi/Recommended-System/blob/master/DIN/train.py#L43