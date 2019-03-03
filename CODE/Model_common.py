# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
模型公共函数

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/3/1 16:22 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import tensorflow as tf
import logging
logger = logging.getLogger('TF')


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


# 计算余弦距离
def tf_cos_dist(tf_np_a, tf_np_b):
    with tf.name_scope('CosDist'):
        # 计算 np_a 的 2范数
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(tf_np_a), axis=1))
        x2_norm = tf.sqrt(tf.reduce_sum(tf.square(tf_np_b), axis=1))

        # 分子
        numerator = tf.reduce_sum(tf.multiply(tf_np_a, tf_np_b), axis=1)
        # 分母  [防止分母为 0 输出为Nan 加上一个极小数 1e-8 ]
        denominator = tf.multiply(x1_norm, x2_norm) + 1e-8

        # 余弦距离
        cos_distance = numerator / denominator

    return cos_distance


# 计算类别的 F1 Score
def tf_f1_score(label_true, label_pred):
    """
    :param label_true:  实际标签值   np.array([1, 1, 2, 0, 1, 1]) 必须为 int 类型
    :param label_pred:  预测标签值   np.array([1, 1, 1, 0, 1, 1]) 必须为 int 类型
    :return:  类别的 Weight权重下 F1 Score [sklearn.metrics.classification_report ]
    """
    with tf.name_scope('F1_Score'):
        # 类别数量
        label_num = tf.cast(tf.reduce_max(label_true)+1, tf.int32)

        # 将实际值转换成 one_hot矩阵
        label_true_array = tf.one_hot(label_true, label_num, dtype=tf.int32)
        # 将预测值转换成 one_hot矩阵
        label_pred_array = tf.one_hot(label_pred, label_num, dtype=tf.int32)

        # 计算各真正例，真反例，假正例，假反例的数量
        tp = tf.count_nonzero(label_pred_array * label_true_array, axis=0)
        # tn = tf.count_nonzero((label_pred_array - 1) * (label_true_array-1), axis=0)  # 真反例
        fp = tf.count_nonzero(label_pred_array * (label_true_array - 1), axis=0)
        fn = tf.count_nonzero((label_pred_array - 1) * label_true_array, axis=0)

        # 转换为浮点型，方便计算
        tp = tf.cast(tp, tf.float64)
        fp = tf.cast(fp, tf.float64)
        fn = tf.cast(fn, tf.float64)

        # 准确率
        precision = tp / (tp + fp + 1e-8)
        # 召回率
        recall = tp / (tp + fn + 1e-8)
        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # 类别数量权重
        weights = tf.reduce_sum(label_true_array, axis=0)
        weights /= tf.reduce_sum(weights)

        # Weight 权重方式修正在总 F1 Score
        f1 = tf.reduce_sum(f1 * weights)

        return f1
