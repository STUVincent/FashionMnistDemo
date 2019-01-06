# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
获取训练、验证 数据

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/1/6 16:47 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import os
import gzip
import numpy as np
from .tf_record import write_tfrecord
from .tf_record import read_tfrecord


# 读取 Fashion-mnist 数据   git@github.com:zalandoresearch/fashion-mnist.git
def load_fashion_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


# 获取测试数据
def get_train_data(args):
    """
    获取训练、测试数据 并返回
    x_data_train, y_data_train, x_data_test, y_data_test
    x_data_*   numpy 格式  (？, 数据维度)
    y_data_*   numpy 格式  (？)     Label 标签（非 OneHot）  int 类型

    :param args:   全局参数字典
    :return:
    """
    args.Logger.info("Get Training Data from : {} ".format(args.trainData__path))

    # 训练数据
    x_data_train, y_data_train = load_fashion_mnist(args.trainData__path, kind='train')

    return x_data_train, y_data_train


# 获取验证数据
def get_dev_data(args):
    """
    获取测试数据 并返回
    x_data_train, y_data_train, x_data_test, y_data_test
    x_data_*   numpy 格式  (？, 数据维度)
    y_data_*   numpy 格式  (？)     Label 标签（非 OneHot）  int 类型

    :param args:   全局参数字典
    :return:
    """
    args.Logger.info("Get Dev Data from : {} ".format(args.trainData__path))

    # 测试数据
    x_data_dev, y_data_dev = load_fashion_mnist(args.trainData__path, kind='t10k')

    return x_data_dev, y_data_dev


# 读取训练数据 TFRecord 数据
def get_train_tfrecord_data(args):

    import pickle

    # 训练数据信息 Pickle 文件
    info_pickle_file = os.path.join(args.TFRecord_path, "{}_info.pickle".format(os.path.basename(args.trainData__path)))
    # 训练数据 转换成 TFRecord 文件保存名称
    train_tfrecord_file = os.path.join(args.TFRecord_path,
                                       "{}_train.tfrecord".format(os.path.basename(args.trainData__path)))

    # 判断训练数据是否已转写入 TFRecord 中， 若未转换则进行转换
    if not os.path.exists(train_tfrecord_file):
        # 读取数据
        x_data_train, y_data_train = get_train_data(args)

        # 训练数据信息
        from collections import Counter
        # 训练数据信息
        train_data_info = {"TrainDataShape": x_data_train.shape,
                           "TrainDataLabel": Counter(y_data_train)}

        # 训练数据信息写入 Pickle 文件 中
        with open(info_pickle_file, 'wb') as f:
            pickle.dump(train_data_info, f)

        from sklearn.preprocessing import OneHotEncoder
        onehot_encoder = OneHotEncoder(dtype=np.int8, n_values=10)
        # 将标签转换成 OneHot 矩阵
        y_data_train = onehot_encoder.fit_transform(y_data_train.reshape(-1, 1)).toarray()

        # 写入 TFRecorder 文件中
        write_tfrecord(x_data_train, y_data_train, tfrecord_file=train_tfrecord_file)

    else:
        args.Logger.info("Get Training Data from : {} ".format(args.TFRecord_path))

        with open(info_pickle_file, 'rb') as f:
            # 读取 训练数据信息
            train_data_info = pickle.load(f)

    # 打印训练数据信息
    args.Logger.info("TrainData Shape:  {}".format(train_data_info['TrainDataShape']))
    args.Logger.info("TrainData Label:  {}".format(train_data_info['TrainDataLabel']))

    # 读取 训练数据 TFRecord
    x_data_train, y_data_train = read_tfrecord(train_tfrecord_file, train_data_info,
                                               epochs=args.trainData__epochs,
                                               batch_size=args.trainData__batch_size,
                                               buffer_size=args.trainData__buffer_size)

    # 训练总次数 加入全局变量中
    args.TrainTime = int(train_data_info['TrainDataShape'][0]*args.trainData__epochs/args.trainData__batch_size)

    return x_data_train, y_data_train
