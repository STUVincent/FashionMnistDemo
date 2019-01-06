# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
写入、读取 数据到 TFRecord 中

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/1/6 16:48 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import tensorflow as tf


# 将数据写入 TFRecord 中
def write_tfrecord(data, labels, tfrecord_file=None):
    """
    :param data:    数据
    :param labels:   标签
    :param tfrecord_file:   保存文件夹路径， 若为 None 则直接保存在当前路径下
    :return:
    """
    if tfrecord_file is None:
        tfrecord_file = 'temp.tfrecord'

    # 创建向TFRecords文件写数据记录的writer
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    # 将数据写入 TFRecord 中
    for i, (data_i, labels_i) in enumerate(zip(data, labels)):
        # 创建example.proto中定义的样例
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    # 'Labels': tf.train.Feature(
                    #     int64_list=tf.train.Int64List(
                    #         value=labels_i)),
                    # 'Labels': tf.train.Feature(
                    #     float_list=tf.train.FloatList(
                    #         value=labels_i)),
                    'Labels': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=labels_i)),
                    'Data': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=data_i.reshape(-1)))}))
        # 将样例序列化为字符串后，写入stat.tfrecord文件
        writer.write(example.SerializeToString())

    # 关闭输出流
    writer.close()


# 提取 TFRecord 文件数据，以 Tensor 格式通过迭代器返回
def read_tfrecord(tfrecord_file, data_info, epochs=1, batch_size=50, buffer_size=1000):
    """
    :param tfrecord_file:   TFRecord 文件 或文件列表
    :param data_info:       训练样本数据信息
    :param epochs:          循环训练迭代次数
    :param batch_size:      每批次样本数据
    :param buffer_size:     随机乱序 buffer 大小
    :return:
    """
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # TODO  需根据实际情况 与 write_tfrecord 的序列样例一致
    # 将一条序列化的样例转换为其包含的所有特征张量
    def parse_example(serial_exmp):
        features = tf.parse_single_example(
            serial_exmp,
            features={
                'Labels': tf.FixedLenFeature([len(data_info['TrainDataLabel'])], tf.int64),
                'Data': tf.FixedLenFeature([data_info['TrainDataShape'][-1]], tf.int64)
            }
        )
        return features['Data'], features['Labels']

    # 从 TFRecord 中提取张量数据
    dataset = dataset.map(parse_example)

    dataset = dataset.repeat(epochs)            # 迭代次数

    if buffer_size > 0:  # 若为 0 则不乱序
        dataset = dataset.shuffle(buffer_size)  # 随机打乱

    dataset = dataset.batch(batch_size)         # 每批数量

    iter_train = dataset.make_one_shot_iterator()

    # 迭代器输出， Tensor 格式，需 sess.run() 转换才获取其数据
    x_data_ops, y_label_ops = iter_train.get_next()

    return x_data_ops, y_label_ops
