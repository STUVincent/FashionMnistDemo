# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
训练数据处理程序
先将训练数据进行清洗后写入 TFRecord 中，再从 TFRecord 中读取送了TensorFlow中训练

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/1/6 16:40 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
from .load_train_data import get_train_tfrecord_data
from .load_train_data import get_dev_data
