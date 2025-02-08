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

from sklearn.metrics import accuracy_score


class ESMM(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation, layer_num):
        super(ESMM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
             for i, feat in enumerate(self.sparse_feature_columns)
        }
        
        self.Dense_ctr = layer.Deep_layer(hidden_units, output_dim, activation)
        self.Dense_cvr = layer.Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[0], inputs[1]
        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)

        ctr_output = self.Dense_ctr(x)
        ctr_output = tf.nn.sigmoid(ctr_output)
        
        cvr_output = self.Dense_cvr(x)
        cvr_output = tf.nn.sigmoid(cvr_output)

        ctcvr_output = ctr_output*cvr_output  # CTCVR = CTR * CVR
        outputs = [ctr_output, ctcvr_output]

        return outputs
    
    
    

if __name__ == '__main__': 
    k = 8
    w_reg = 1e-4
    b_reg = 1e-4
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    layer_num = 4
    input_dim = 10
    expert_dim = 20
    n_expert = 4
    n_task = 2
    output_dims = [1] * n_task
    num_samples = 1000
    batch_size = 32
    epochs = 10
    
    
    [feature_columns,y_train,y_test,
    x_train_categ, x_test_categ,
    x_train_oht,x_test_oht,
    x_train_cross,x_test_cross,
    x_train_conti, x_test_conti] = datasetProcess.loadDataset("adult")
    
    
    ##input:[x_train_conti,x_train_categ]                                   
    X_train = (x_train_conti.values,x_train_categ.values)
    X_test = (x_test_conti.values,x_test_categ.values)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (y_train,y_train)))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        
    model = ESMM(feature_columns, hidden_units, output_dim, activation, layer_num)
    optimizer = optimizers.SGD(0.01)
    
    
    losses = ['binary_crossentropy'] * n_task
    # 修改 metrics 参数，使其长度与模型输出数量一致
    metrics = ['accuracy'] * n_task        
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    model.fit(train_dataset, epochs=2)
    #[logloss, auc = model.evaluate(X_test, (y_test,y_test))
    #print('logloss {}\nAUC {}'.format(round(logloss,2), round(auc,2)))
    model.summary()
