# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
神经网络基础层
所有的正则损失默认存储在 losses 变量空间中

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/1/6 16:42 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import tensorflow as tf
import logging
logger = logging.getLogger('TensorFlow')


# 通过 tf.get_variable 函数来获取权重变量。
def get_weight_variable(shape, regularizer=None):
    """
    :param shape:  当前层权重变量大小
    :param regularizer:  正则化
                        L2 正则:  tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
                        L1 正则:  tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)
                        None : 不加入参数正则化
    :return:  返回初始化的权重变量
    """
    # 当前权重初始化设置
    weights = tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 判断是否有正则化，若有则将损失值加入 集合losses 中
    if regularizer is not None:
        with tf.name_scope('W'):
            tf.add_to_collection('losses', regularizer(weights))
    # 返回生成的权重变量
    return weights


# 通过 get_bias_variable 函数来获取偏置值变量
def get_bias_variable(shape, regularizer=None):
    """
    :param shape:  当前层权重变量大小
    :param regularizer:  正则化
                        L2 正则:  tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
                        L1 正则:  tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)
                        None : 不加入参数正则化
    :return:  返回初始化的权重变量
    """
    # 当前层偏置值变量
    biases = tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.1))

    # 判断是否有正则化，若有则将损失值加入 集合losses 中 （一般偏置值不会进行参数正则化）
    if regularizer is not None:
        with tf.name_scope('B'):
            tf.add_to_collection('losses', regularizer(biases))

    # 返回生成的权重变量
    return biases


# 创建神经网络 卷积层
def create_convolution_layer(input_tensor, parameters):
    """
    :param input_tensor:  输入张量
    :param parameters:    参数字典
                    layer:   当前层名称
                    fsize:  [卷积核大小, 卷积核大小, 上一层深度, 下一层深度]
                    strides:   [固定值 1，步长，步长，固定值 1]
                    padding:  ['SAME':表示添加全 0 填充。  'VALID':  表示不添加]
                    regularizer_w:  权重 正则化
                    regularizer_b:  偏置值 正则化
                    activation:  激活函数  relu /sigmoid/ tanh
    :return:  output_tensor
    """
    with tf.variable_scope('Layer%s_Conv' % parameters['layer']):
        weights = get_weight_variable(parameters['fsize'], regularizer=parameters['regularizer_w'])

        biases = get_bias_variable(parameters['fsize'][3], regularizer=parameters['regularizer_b'])

        output_conv = tf.nn.conv2d(input_tensor, weights, strides=parameters['strides'], padding=parameters['padding'])
        output_tensor = tf.nn.bias_add(output_conv, biases)

        # 增加激活函数
        if parameters['activation'] == 'relu':
            output_tensor = tf.nn.relu(output_tensor)
        elif parameters['activation'] == 'sigmoid':
            output_tensor = tf.nn.sigmoid(output_tensor)
        elif parameters['activation'] == 'tanh':
            output_tensor = tf.nn.tanh(output_tensor)
        # 非法值
        else:
            output_tensor = input_tensor
            logger.error('Error, conv method [{:s}] wrong! please input "relu | sigmoid | tanh"'.
                         format(parameters['activation']))
            raise Exception('Error, conv method [%s] wrong! please input "relu | sigmoid | tanh"' %
                            parameters['activation'])

        logger.info('Layer:{} Conv  {}'.format(parameters['layer'], output_tensor.get_shape()))
        logger.info('                              activation:{} fsize:{} strides:{} padding:{}'.format(
            parameters['activation'], parameters['fsize'], parameters['strides'], parameters['padding']))

    return output_tensor


# 创建神经网络 池化层
def create_pooling_layer(input_tensor, parameters):
    """
    :param input_tensor:  输入张量
    :param parameters:    参数字典
                        layer:   当前层名称
                        ksize:    池化核大小  [固定值 1，池化层大小，池化层大小，固定值 1]
                        strides:  步长   [固定值 1，步长，步长，固定值 1]
                        padding:  ['SAME':表示添加全 0 填充。  'VALID':  表示不添加]
                        method:   池化方式  max/avg
    :return: output_tensor
    """
    with tf.variable_scope('Layer%s_Pool' % parameters['layer']):
        # 最大值池化
        if parameters['method'] == 'max':
            output_tensor = tf.nn.max_pool(input_tensor, ksize=parameters['ksize'], strides=parameters['strides'],
                                           padding=parameters['padding'])
        # 平均值池化
        elif parameters['method'] == 'avg':
            output_tensor = tf.nn.avg_pool(input_tensor, ksize=parameters['ksize'], strides=parameters['strides'],
                                           padding=parameters['padding'])
        # 非法值
        else:
            output_tensor = input_tensor
            logger.error('Error, pool method [{:s}] wrong! please input "max" or "avg"'.format(parameters['method']))
            raise Exception('Error, pool method [%s] wrong! please input "max" or "avg"' % parameters['method'])

    logger.info('Layer:{} Pool  {}'.format(parameters['layer'], output_tensor.get_shape()))
    logger.info('                              method:{} ksize:{} strides:{} padding:{}'.format(
        parameters['method'], parameters['ksize'], parameters['strides'], parameters['padding']))

    return output_tensor


# 创建神经网络 全连接层
def create_fullconnection_layer(input_tensor, parameters):
    """
    :param input_tensor: 输入张量
    :param parameters:  参数字典
                         layer:   当前层名称
                         fcsize:   当前层的设置参数
                         regularizer_w:  权重 正则化
                         regularizer_b:  偏置值 正则化
                         activation:  激活函数  relu /sigmoid/ tanh
                         dropout_keep_p: dropout 函数 张量保存概率 参数范围[0，1]  为 None 时 不加入dropout
                                       dropout 一般只用在全连接层而不是卷积层和池化层
    :return:  output_tensor
    """
    with tf.variable_scope('Layer%s_Full' % parameters['layer']):
        weights = get_weight_variable([input_tensor.get_shape()[-1].value, parameters['fcsize']],
                                      regularizer=parameters['regularizer_w'])

        biases = get_bias_variable([parameters['fcsize']], regularizer=parameters['regularizer_b'])

        output_tensor = tf.matmul(input_tensor, weights) + biases

        # 增加激活函数
        if parameters['activation'] == 'relu':
            output_tensor = tf.nn.relu(output_tensor)
        elif parameters['activation'] == 'sigmoid':
            output_tensor = tf.nn.sigmoid(output_tensor)
        elif parameters['activation'] == 'tanh':
            output_tensor = tf.nn.tanh(output_tensor)
        # 非法值
        else:
            output_tensor = input_tensor
            logger.error('Error, Activation Function [{:s}] wrong! please input "relu | sigmoid | tanh"'.
                         format(parameters['activation']))
            raise Exception('Error, Activation Function [%s] wrong! please input "relu | sigmoid | tanh"' %
                            parameters['activation'])

        # 设置为非 None 时，调用 dropout 函数
        if parameters['dropout_keep_p'] is not None:
            # 调用 dropout 函数
            output_tensor = tf.nn.dropout(output_tensor, parameters['dropout_keep_p'], name='dropout')

    logger.info('Layer:{} Full  {}'.format(parameters['layer'], output_tensor.get_shape()))
    logger.info('                              activation:{} fcsize:{} dropout_keep_p:{}'.format(
        parameters['activation'], parameters['fcsize'], parameters['dropout_keep_p']))

    return output_tensor


# 输出层
def create_output_layer(input_tensor, output_length):
    """
    :param input_tensor: 输入张量
    :param output_length:  输出层维度
    :return:  output_tensor
    """
    with tf.variable_scope('Layer_Output'):
        # 当前层权重变量
        weights = tf.get_variable("weight", [input_tensor.get_shape()[-1].value, output_length],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 当前层偏置值变量
        biases = tf.get_variable("bias", output_length, initializer=tf.constant_initializer(0.1))

        output_tensor = tf.matmul(input_tensor, weights) + biases

        # 增加激活函数 [输出层一般使用 sigmoid 激活函数]
        output_tensor = tf.nn.sigmoid(output_tensor)

    logger.info('OutputLayer    {}\n'.format(output_tensor.get_shape()))

    return output_tensor
