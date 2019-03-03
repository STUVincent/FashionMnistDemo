# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
# 当前项目: FashionMnist_Demo
# 创建时间: 2019/3/1 11:19 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import tensorflow as tf
from CODE.Model_common import create_fullconnection_layer
from CODE.Model_common import create_output_layer
from CODE.Model_common import tf_f1_score


# BP 神经网络
def inference(logger, input_length, output_length, l1_scale, l2_scale, label_weight=None):
    """
    :param logger:         日志对象
    :param input_length:   输入层维度
    :param output_length:  输出层维度
    :param l1_scale:       L1 正则权重系数
    :param l2_scale:       L2 正则权重系数
    :param label_weight:  类别权重 [解决数据类别不均衡问题] 为None时，默认权重都为1  # [100, 1, 100]
     """
    # 定义 输入 输出 dropout_keep_p  placeholder
    with tf.variable_scope('input'):
        input_x = tf.placeholder(tf.float32, [None, input_length], name='input_x')
        input_y = tf.placeholder(tf.float32, [None, output_length], name='input_y')
        dropout_keep_p = tf.placeholder(tf.float32, name="dropout_keep_p")
    
    # L1 正则
    if l1_scale is None:
        regular_l1 = None
    else:
        regular_l1 = tf.contrib.layers.l1_regularizer(l1_scale)
    
    # L2 正则
    if l2_scale is None:
        regular_l2 = None
    else:
        regular_l2 = tf.contrib.layers.l2_regularizer(l2_scale)
    
    # ################################################################################
    logger.info('Creating a BP Neural Network Model。   Tensorflow Version :{}  \n'.format(tf.__version__))
    logger.info('[input]      {}'.format(input_x.get_shape()))
    
    # 第1层 全连接层
    parameters = {'layer': '01',  # 当前层序号
                  'fcsize': 1000,  # 当前层节点数
                  'regularizer_w': regular_l1,  # 权重 正则化 （None 为不使用正则化）
                  'regularizer_b': regular_l2,  # 偏置值 正则化 （None 为不使用正则化）
                  'activation': 'relu',  # 激活函数 relu sigmoid tanh
                  'dropout_keep_p': dropout_keep_p  # dropout 函数 张量保存概率 参数范围[0，1]
                  }
    output_tensor = create_fullconnection_layer(input_x, parameters)
    
    # 第2层 全连接层
    parameters = {'layer': '02',  # 当前层序号
                  'fcsize': 100,  # 当前层节点数
                  'regularizer_w': regular_l1,  # 权重 正则化 （None 为不使用正则化）
                  'regularizer_b': regular_l2,  # 偏置值 正则化 （None 为不使用正则化）
                  'activation': 'relu',  # 激活函数 relu sigmoid tanh
                  'dropout_keep_p': dropout_keep_p  # dropout 函数 张量保存概率 参数范围[0，1]
                  }
    output_tensor = create_fullconnection_layer(output_tensor, parameters)
    
    # 输出层
    output_tensor = create_output_layer(output_tensor, output_length)
    
    # ################################################################################
    # 变量监测
    # tf.add_to_collection('Tensorboard', output_tensor)
    # tf.add_to_collection('Tensorboard', tf.get_default_graph().get_tensor_by_name('Layer01_Full/bias:0'))
    
    # 判断 模型输出 与 y 维度大小 是否一致，若不一致 报错退出程序
    assert str(input_y.shape) == str(output_tensor.shape), \
        logger.error('Error： 模型输出层维度大小 错误！！！...... ')
    # ################################################################################
    with tf.variable_scope('output'):
        # 计算 softmax 后概率 (每个类别的预测概率)
        predict_p = tf.nn.softmax(output_tensor, name='predict_p')
        # 预测值
        predict = tf.argmax(predict_p, 1, name="predict")
    
    # 损失函数
    with tf.name_scope('loss'):
        # TODO 加入权重以解决数据不均衡问题
        if label_weight is None:  # 若label_weight为None，即每个类别权重都为 1
            label_weight = [1] * input_y.shape[-1]
    
        # 累计权重矩阵
        weight = 0
        for label_i, weight_i in enumerate(label_weight):
            weight += weight_i * input_y[:, label_i]
    
        # 加权交叉熵  Cross_entropy 分类
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=input_y, logits=predict_p) * weight, name='cross_entropy_loss')
    
        # 所有损失函数叠加
        tf.add_to_collection('losses', cross_entropy_loss)
        loss = tf.add_n(tf.get_collection('losses'))
    
    # 准确率 TODO 考虑加入 F1 等其它指标  tf.metrics
    with tf.name_scope("accuracy"):
        correct_predict = tf.equal(tf.argmax(input_y, 1), predict)
        accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"), name="accuracy")
    
    # F1 Score
    with tf.name_scope("F1_score"):
        f1_score = tf_f1_score(tf.argmax(input_y, 1), predict)

    # ###################################### 打印模型接口参数
    logger.info('input  Tensor：{:15s}  {}   '.format('input_x', input_x))
    logger.info('input  Tensor：{:15s}  {}   '.format('input_y', input_y))
    logger.info('input  Tensor：{:15s}  {}   '.format('dropout_keep_p', dropout_keep_p))
    logger.info('output Tensor：{:15s}  {}   '.format('predict', predict))
    logger.info('output Tensor：{:15s}  {}   '.format('predict_p', predict_p))

    model = {"input_x": input_x,
             "input_y": input_y,
             "dropout_keep_p": dropout_keep_p,
             "predict": predict,
             "predict_p": predict_p,
             "loss": loss,
             "accuracy": accuracy,
             "f1_score": f1_score,
             }
    return model
