# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
# 当前项目: FashionMnist_Demo
# 创建时间: 2019/3/1 21:55 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import tensorflow as tf
from CODE.Model_common import create_fullconnection_layer
from CODE.Model_common import create_output_layer
from CODE.Model_common import tf_f1_score
from CODE.Model_common import create_convolution_layer
from CODE.Model_common import create_pooling_layer


# CNN 卷积神经网络
def inference(logger, input_length, output_length, l1_scale=None, l2_scale=None, label_weight=None):
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

    image_size = 28  # 图片大小
    num_channels = 1  # 图片颜色通道

    f_size_list = [3, 4, 5]  # 每个卷积核大小
    num_filters = 5  # 卷积核个数

    # 判断 模型输入 x  是否能转换成指定维度
    assert input_length % (image_size*image_size*num_channels) == 0, \
        logger.error('Error： 模型输入层维度转换失败！！！...... ')
    # ################################################################################
    logger.info('Creating a CNN Neural Network Model。   Tensorflow Version :{}  \n'.format(tf.__version__))
    logger.info('[input]      {}'.format(input_x.get_shape()))

    # 将训练数据 x 转换到指定维度
    with tf.variable_scope('Reshape'):
        output_tensor = tf.reshape(input_x, (-1, image_size, image_size, num_channels))
        logger.info('Reshape        {}'.format(output_tensor.get_shape()))

    # 第一层 卷积核组合层输出　tensor
    output_tensor_concat = []

    for f_i, f_size in enumerate(f_size_list):
        # 卷积
        parameters = {'layer': '1{}'.format(f_i),  # 当前层序号
                      'fsize': [f_size, f_size, output_tensor.get_shape()[-1].value, num_filters],
                      # [卷积核大小, 卷积核大小, 上一层深度, 下一层深度]
                      'strides': [1, 1, 1, 1],  # [固定值 1，步长，步长，固定值 1]
                      'padding': 'SAME',  # ['SAME':表示添加全 0 填充。  'VALID':  表示不添加]
                      'regularizer_w': None,  # 权重 正则化
                      'regularizer_b': None,  # 偏置值 正则化
                      'activation': 'relu'  # 激活函数  relu /sigmoid/ tanh
                      }
        output_tensor_conv = create_convolution_layer(output_tensor, parameters)
        # 池化
        parameters = {'layer': '1{}'.format(f_i),  # 当前层序号
                      'ksize': [1, 2, 2, 1],  # [固定值 1，池化层大小，池化层大小，固定值 1]
                      'strides': [1, 2, 2, 1],  # [固定值 1，步长，步长，固定值 1]
                      'padding': 'SAME',  # ['SAME':表示添加全 0 填充。  'VALID':  表示不添加]
                      'method': 'max'  # 池化方式  max/avg
                      }
        output_tensor_pool = create_pooling_layer(output_tensor_conv, parameters)

        output_tensor_concat.append(output_tensor_pool)

    # 将 N个卷积 池化后矩阵拉合并，并伸为一个向量 以便后续全连接计算
    with tf.name_scope('Concat'):
        # 合并
        output_tensor = tf.concat(output_tensor_concat, 3)
        logger.info('Concat         {}'.format(output_tensor.get_shape()))

    # 第二层 卷积层
    parameters = {'layer': '20',  # 当前层序号
                  'fsize': [3, 3, output_tensor.get_shape()[-1].value, 8],  # [卷积核大小, 卷积核大小, 上一层深度, 下一层深度]
                  'strides': [1, 1, 1, 1],  # [固定值 1，步长，步长，固定值 1]
                  'padding': 'SAME',  # ['SAME':表示添加全 0 填充。  'VALID':  表示不添加]
                  'regularizer_w': None,  # 权重 正则化
                  'regularizer_b': None,  # 偏置值 正则化
                  'activation': 'relu'  # 激活函数  relu /sigmoid/ tanh
                  }
    output_tensor = create_convolution_layer(output_tensor, parameters)

    # 第三层 池化层
    parameters = {'layer': '30',  # 当前层序号
                  'ksize': [1, 2, 2, 1],  # [固定值 1，池化层大小，池化层大小，固定值 1]
                  'strides': [1, 2, 2, 1],  # [固定值 1，步长，步长，固定值 1]
                  'padding': 'VALID',  # ['SAME':表示添加全 0 填充。  'VALID':  表示不添加]
                  'method': 'max'  # 池化方式  max/avg
                  }
    output_tensor = create_pooling_layer(output_tensor, parameters)

    # 将 矩阵拉伸为一个向量 以便后续全连接计算
    with tf.name_scope('Reshape'):
        output_shape = output_tensor.get_shape().as_list()
        nodes = output_shape[1] * output_shape[2] * output_shape[3]
        output_tensor = tf.reshape(output_tensor, [-1, nodes])
        logger.info('Reshape        {}'.format(output_tensor.get_shape()))

    # Dropout 层
    with tf.name_scope("Dropout"):
        output_tensor = tf.nn.dropout(output_tensor, dropout_keep_p)
        logger.info('Dropout        {}'.format(output_tensor.get_shape()))

    # 第四层 全连接层
    parameters = {'layer': '40',   # 当前层序号
                  'fcsize': 20,    # 当前层节点数
                  'regularizer_w': regular_l1,   # 权重 正则化 （None 为不使用正则化）
                  'regularizer_b': regular_l2,    # 偏置值 正则化 （None 为不使用正则化）
                  'activation': 'relu',   # 激活函数 relu sigmoid tanh
                  'dropout_keep_p': dropout_keep_p  # dropout 函数 张量保存概率 参数范围[0，1]
                  }
    output_tensor = create_fullconnection_layer(output_tensor, parameters)

    # 输出层
    output_tensor = create_output_layer(output_tensor, output_length)

    # ################################################################################
    # 变量监测
    tf.add_to_collection('Tensorboard', output_tensor)
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

    # tf_f1_score
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
