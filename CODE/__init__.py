# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
# 当前项目: FashionMnist_Demo
# 创建时间: 2019/1/6 16:40 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
from .train_data import get_train_tfrecord_data   # 获取训练数据 TFRecord 数据
from .train_data import get_dev_data              # 获取验证数据
from .model import BP, CNN, CNN2                  # 创建模型
from .model import run_trainer                    # 训练模型
