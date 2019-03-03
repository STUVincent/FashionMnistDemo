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
from .Configure import GlobalConstants            # 全局常量
from .Train_common import print_model_info        # 打印模型训练变量、 损失函数、 TensorBoard 监控变量
from .Train_common import tensorboard_monitor     # Tensorboard 监测
from .Train_common import train_optimizer         # 优化器
from .Init import argument_init                   # 输入参数设置、全局参数、日志初始化
from .Train_common import save_variables_and_metagraph  # 模型训练过程保存
