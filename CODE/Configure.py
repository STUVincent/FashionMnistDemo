# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
全局常量： 项目中需要使用的全局变量都将保存在此类中

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/3/1 10:46 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""


# 全局常量设置
class GlobalConstants(object):
    # 输入模型 X 数据维度
    model_x_shape = 784
    # 输出模型 Y 类别维度
    model_y_shape = 10

    # 待加入 TensorBoard 监控的 Tensor 名称 【此参数可为空，但不能删除此变量】
    monitor_tensor = ['Layer01_Full/bias:0', 'input/dropout_keep_p:10', 'Concat/concat:0', 'Layer_Output/weight:0']
