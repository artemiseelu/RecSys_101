# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:38:04 2025

@author: emmalu
"""


import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Flatten, Layer, Embedding
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2


class Wide_layer(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        """
        构建层的可训练权重。
        Build the trainable weights of the layer.

        参数说明：
        Parameters:
        - input_shape: 输入张量的形状。The shape of the input tensor.

        初始化两个可训练参数：
        Initialize two trainable parameters:
        - w0: 偏置项，形状为(1, 1)。Bias term with shape (1, 1).
        - w: 权重矩阵，形状为(input_shape[-1], 1)。Weight matrix with shape (input_shape[-1], 1).
        """
        ##para: bias w0, weight w
        # 初始化偏置项 w0，初始值为0，可训练
        # Initialize the bias term w0 with an initial value of 0, which is trainable
        self.w0 = self.add_weight(name='w0', shape=(1, 1),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        # 初始化权重矩阵 w，使用随机正态分布初始化，可训练，并使用L2正则化
        # Initialize the weight matrix w with a random normal distribution, which is trainable and uses L2 regularization
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(1e-4))

    def call(self, inputs, **kwargs):
        """
        前向传播计算。
        Forward propagation calculation.

        参数说明：
        Parameters:
        - inputs: 输入张量。The input tensor.
        - **kwargs: 其他关键字参数。Other keyword arguments.

        返回值：
        Returns:
        - output: 输出张量，形状为(batchsize, 1)。The output tensor with shape (batchsize, 1).
        """
        # wide layer：w^T * x + w0
        # 计算输入与权重矩阵的乘积，并加上偏置项
        # Calculate the product of the input and the weight matrix, and then add the bias term
        output = tf.matmul(inputs, self.w) + self.w0  # shape: (batchsize, 1)
        return output


class Deep_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        """
        初始化深度层。
        Initialize the deep layer.

        :param hidden_units: 一个列表，包含每个隐藏层的神经元数量。
                             A list containing the number of neurons in each hidden layer.
        :param output_dim: 输出层的维度。
                           The dimension of the output layer.
        :param activation: 隐藏层使用的激活函数。
                           The activation function used in the hidden layers.
        """
        super().__init__()
        # 创建隐藏层列表，每个隐藏层是一个全连接层
        # Create a list of hidden layers, where each hidden layer is a fully connected layer
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        # 创建输出层，不使用激活函数
        # Create the output layer without using an activation function
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs, **kwargs):
        """
        前向传播计算。
        Forward propagation calculation.

        :param inputs: 输入张量。
                       The input tensor.
        :param kwargs: 其他关键字参数。
                       Other keyword arguments.
        :return: 输出张量，形状为 (batchsize, output_dim)。
                 The output tensor with shape (batchsize, output_dim).

        网络结构：
        Network structure:
        Dense1(hidden_units1) ->
        Dense2(hidden_units2) ->
        Dense3(hidden_units2) ->
        Dense_output(output_dim)

        ###TODO: 是否增加 BN、dropout 等操作
        ###TODO: Whether to add operations such as BN, dropout
        """
        x = inputs
        # 依次通过每个隐藏层
        # Pass through each hidden layer in turn
        for layer in self.hidden_layer:
            x = layer(x)
        # 通过输出层得到最终输出
        # Pass through the output layer to get the final output
        output = self.output_layer(x)  # shape: (batchsize, output_dim)
        return output



class FM_layer(Layer):
    def __init__(self, k, w_reg, v_reg):
        """
        初始化FM层。
        Initialize the FM (Factorization Machines) layer.

        :param k: 隐向量的维度，用于二阶交互部分。
                  The dimension of the latent vector for the second-order interaction part.
        :param w_reg: 一阶权重的L2正则化系数。
                      The L2 regularization coefficient for the first-order weights.
        :param v_reg: 二阶隐向量的L2正则化系数。
                      The L2 regularization coefficient for the second-order latent vectors.
        """
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        """
        构建层的可训练权重。
        Build the trainable weights of the layer.

        :param input_shape: 输入张量的形状。
                            The shape of the input tensor.

        初始化三个可训练参数：
        Initialize three trainable parameters:
        - w0: 全局偏置项，形状为(1, 1)。
              The global bias term with shape (1, 1).
        - w: 一阶权重矩阵，形状为(input_shape[-1], 1)。
             The first-order weight matrix with shape (input_shape[-1], 1).
        - v: 二阶隐向量矩阵，形状为(input_shape[-1], k)。
             The second-order latent vector matrix with shape (input_shape[-1], k).
        """
        ##para: bias w0; weight w (for 1st order); weight v (for 2nd order)
        # 初始化全局偏置项 w0，初始值为0，可训练
        # Initialize the global bias term w0 with an initial value of 0, which is trainable
        self.w0 = self.add_weight(name='w0', shape=(1, 1),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        # 初始化一阶权重矩阵 w，使用随机正态分布初始化，可训练，并应用L2正则化
        # Initialize the first-order weight matrix w with a random normal distribution, which is trainable and applies L2 regularization
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        # 初始化二阶隐向量矩阵 v，使用随机正态分布初始化，可训练，并应用L2正则化
        # Initialize the second-order latent vector matrix v with a random normal distribution, which is trainable and applies L2 regularization
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),  # v(i,k)
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        """
        前向传播计算。
        Forward propagation calculation.

        :param inputs: 输入张量。
                       The input tensor.
        :param kwargs: 其他关键字参数。
                       Other keyword arguments.
        :return: 输出张量，形状为(batchsize, 1)。
                 The output tensor with shape (batchsize, 1).

        计算包括两部分：
        The calculation includes two parts:
        - 线性部分（与Wide层相同）：w^T * x + w0
          The linear part (same as the Wide layer): w^T * x + w0
        - 二阶交互部分：0.5 * (sum(pow(v^T * inputs), 2) - (pow(v, 2)^T * pow(inputs, 2)))
          The second-order interaction part: 0.5 * (sum(pow(v^T * inputs), 2) - (pow(v, 2)^T * pow(inputs, 2)))
        """
        # 检查输入维度是否为2维
        # Check if the input has 2 dimensions
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))
        # 计算线性部分，等同于Wide层的计算
        # Calculate the linear part, which is the same as the calculation in the Wide layer
        # linear_part = layer.Wide_layer(inputs)
        linear_part = tf.matmul(inputs, self.w) + self.w0  # shape: (batchsize, 1)
        # 计算二阶交互部分的第一项：(v^T * inputs)^2
        # Calculate the first term of the second-order interaction part: (v^T * inputs)^2
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  # shape: (batchsize, self.k)
        # 计算二阶交互部分的第二项：(v^2)^T * (inputs^2)
        # Calculate the second term of the second-order interaction part: (v^2)^T * (inputs^2)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))  # shape: (batchsize, self.k)
        # 计算二阶交互部分，对两项的差求和并乘以0.5
        # Calculate the second-order interaction part by summing the difference between the two terms and multiplying by 0.5
        inter_part = 0.5 * tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True)  # shape: (batchsize, 1)

        # 将线性部分和二阶交互部分相加得到最终输出
        # Add the linear part and the second-order interaction part to get the final output
        output = linear_part + inter_part
        return output

class FFM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        """
        初始化FFM层。
        Initialize the FFM (Field-aware Factorization Machines) layer.

        :param feature_columns: 特征列信息，包含稠密特征列和稀疏特征列
        :param feature_columns: Information of feature columns, including dense feature columns and sparse feature columns
        :param k: 隐向量的维度
        :param k: Dimension of the latent vector
        :param w_reg: 线性权重的正则化系数
        :param w_reg: Regularization coefficient for linear weights
        :param v_reg: 隐向量的正则化系数
        :param v_reg: Regularization coefficient for latent vectors
        """
        super(FFM_Layer, self).__init__()
        # 拆分特征列信息为稠密特征列和稀疏特征列
        # Split the feature column information into dense and sparse feature columns
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        # 计算特征总数
        # Calculate the total number of features
        self.feature_num = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) + len(self.dense_feature_columns)
        # 计算场的数量
        # Calculate the number of fields
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)

    def build(self, input_shape):
        """
        构建层的可训练权重。
        Build the trainable weights of the layer.
        """
        # 初始化偏置项
        # Initialize the bias term
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        # 初始化线性权重
        # Initialize the linear weights
        self.w = self.add_weight(name='w', shape=(self.feature_num, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        # 初始化隐向量 shape:[feature_num,field_num,k]
        # Initialize the latent vectors
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_num, self.field_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        """
        前向传播计算。
        Forward propagation calculation.

        :param inputs: 输入张量
        :param inputs: Input tensor
        :return: 输出张量
        :return: Output tensor
        """
        # 计算线性部分，公式为 w0 + w^T * x
        # Calculate the linear part using the formula w0 + w^T * x
        linear_part = self.w0 + tf.matmul(inputs, self.w)

        # 每维特征先跟自己的 [field_num, k] 相乘得到Vij*X
        # Multiply each feature dimension with its corresponding [field_num, k] matrix to get Vij*X
        field_f = tf.tensordot(inputs, self.v, axes=1)
 
        # # 域之间两两相乘，
        # v1循环版本
        # 此部分代码使用双重循环计算交叉项，效率较低，已注释掉
        # This part of the code calculates the interaction terms using double loops, which is less efficient and has been commented out
        # for i in range(self.field_num):
        #     for j in range(i+1, self.field_num):
        #         inter_part += tf.reduce_sum(
        #             tf.multiply(field_f[:, i], field_f[:, j]), # [None, 8]
        #             axis=1, keepdims=True
        #         )    
        
        # # 域之间两两相乘，
        # einsum版本
        # 计算所有场之间的外积
        # Calculate the outer product between all fields
        outer_product = tf.einsum('...ik,...jk->...ij', field_f, field_f)
        # 提取上三角矩阵（不包含对角线）
        # Extract the upper triangular matrix (excluding the diagonal)
        upper_triangular_mask = tf.linalg.band_part(tf.ones((self.field_num, self.field_num)), 0, -1) - tf.eye(self.field_num)
        upper_triangular = outer_product * upper_triangular_mask
        # 对每个样本的上三角矩阵求和得到交叉项
        # Sum the upper triangular matrix for each sample to get the interaction terms
        inter_part = tf.reduce_sum(upper_triangular, axis=[-2, -1], keepdims=True)
        #print(inter_part.shape)
        # 获得batch大小，直接使用shape[0]会产生None值
        # Get the batch size. Using shape[0] directly may result in a None value
        batch_size = tf.shape(inter_part)[0]
        # 调整交叉项的形状为 [batch_size, 1]
        # Reshape the interaction terms to [batch_size, 1]
        inter_part = tf.reshape(inter_part, [batch_size, 1])

        # 合并线性部分和交叉项
        # Combine the linear part and the interaction terms
        output = linear_part + inter_part

        return output   




class Cross_layer(Layer):
    def __init__(self, layer_num, w_reg=1e-4, b_reg=1e-4):
        super().__init__()
        self.layer_num = layer_num
        self.w_reg = w_reg
        self.b_reg = b_reg

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w'+str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.w_reg),
                            trainable=True)
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b'+str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.b_reg),
                            trainable=True)
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)  # (None, dim, 1)
        xl = x0  # (None, dim, 1)
        for i in range(self.layer_num):
            # 先乘后两项（忽略第一维，(dim, 1)表示一个样本的特征）
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.cross_weight[i]) # (None, 1, 1)
            # # 乘x0，再加上b、xl
            xl = tf.matmul(x0, xl_w) + self.cross_bias[i] + xl  # (None, dim, 1)

        output = tf.squeeze(xl, axis=2)  # (None, dim)
        return output


class MMoE_Layer(tf.keras.layers.Layer):
    def __init__(self, expert_dim, n_expert, n_task, hidden_units, activation):
        super(MMoE_Layer, self).__init__()
        self.n_task = n_task
        # 使用 Deep_layer 构建专家网络
        #self.expert_layer = [Deep_layer(hidden_units, expert_dim, activation) for i in range(n_expert)]
        self.expert_layer = [Dense(expert_dim, activation='relu') for i in range(n_expert)]
        # 使用 Deep_layer 构建门控网络，注意门控网络输出维度为 n_expert 且激活函数为 softmax
        #self.gate_layers = [Deep_layer(hidden_units, n_expert, 'softmax') for i in range(n_task)]
        self.gate_layers = [Dense(n_expert, activation='softmax') for i in range(n_task)]
        
    def call(self, x):
        # 构建多个专家网络
        E_net = [expert(x) for expert in self.expert_layer]
        E_net = Concatenate(axis=1)([e[:, tf.newaxis, :] for e in E_net])  # 维度 (bs,n_expert,n_dims)
        # 构建多个门网络
        gate_net = [gate(x) for gate in self.gate_layers]  # 维度 n_task 个 (bs,n_expert)
        # towers 计算：对应的门网络乘上所有的专家网络
        towers = []
        for i in range(self.n_task):
            g = tf.expand_dims(gate_net[i], axis=-1)  # 维度(bs,n_expert,1)
            _tower = tf.matmul(E_net, g, transpose_a=True)
            towers.append(Flatten()(_tower))  # 维度(bs,expert_dim)
        return towers
    
    
class Interaction_layer(Layer):
    '''
    # input shape:  [None, field, k]
    # output shape: [None, field*(field-1)/2, k]
    '''
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs): # [None, field, k]
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        element_wise_product_list = []
        for i in range(inputs.shape[1]):
            for j in range(i+1, inputs.shape[1]):
                element_wise_product_list.append(tf.multiply(inputs[:, i], inputs[:, j]))  #[t, None, k]
        element_wise_product = tf.transpose(tf.convert_to_tensor(element_wise_product_list), [1, 0, 2]) #[None, t, k]
        return element_wise_product

class Attention_layer(Layer):
    '''
    # input shape:  [None, n, k]
    # output shape: [None, k]
    '''
    def __init__(self):
        super().__init__()

    def build(self, input_shape): # [None, field, k]
        self.attention_w = Dense(input_shape[1], activation='relu')
        self.attention_h = Dense(1, activation=None)

    def call(self, inputs, **kwargs): # [None, field, k]
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        x = self.attention_w(inputs)  # [None, field, field]
        x = self.attention_h(x)       # [None, field, 1]
        a_score = tf.nn.softmax(x)
        a_score = tf.transpose(a_score, [0, 2, 1]) # [None, 1, field]
        output = tf.reshape(tf.matmul(a_score, inputs), shape=(-1, inputs.shape[2]))  # (None, k)
        return output

class AFM_layer(Layer):
    def __init__(self, feature_columns, mode):
        super(AFM_layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layer = {"emb_"+str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                            for i, feat in enumerate(self.sparse_feature_columns)}
        self.interaction_layer = Interaction_layer()
        if self.mode=='att':
            self.attention_layer = Attention_layer()
        self.output_layer = Dense(1)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        embed = [self.embed_layer['emb_'+str(i)](sparse_inputs[:, i])
               for i in range(sparse_inputs.shape[1])]  # list
        embed = tf.convert_to_tensor(embed)
        embed = tf.transpose(embed, [1, 0, 2])  #[None, 26，k]

        # Pair-wise Interaction
        embed = self.interaction_layer(embed)

        if self.mode == 'avg':
            x = tf.reduce_mean(embed, axis=1)  # (None, k)
        elif self.mode == 'max':
            x = tf.reduce_max(embed, axis=1)  # (None, k)
        else:
            x = self.attention_layer(embed)  # (None, k)

        output = tf.nn.sigmoid(self.output_layer(x))
        return output