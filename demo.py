# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
以 FashionMnist 数据集进行的 TensorFlow Demo程序

FashionMnist 数据来源
git@github.com:zalandoresearch/fashion-mnist.git

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/1/6 16:51 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5  TensorFlow 1.12.0
# 版    本: V1.0
"""
import argparse
import logging
import os
import time
import tensorflow as tf
from CODE import get_train_tfrecord_data   # 获取训练数据 TFRecord 形式
from CODE.LOG import logger_set            # 获取日志设置函数

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '  # 只输出error


# 输入参数设置、全局参数、日志初始化
def argument_init():
    # ############################ CMD 传参 ################################################
    parser = argparse.ArgumentParser(
        description="""
        TensorFlow Demo 代码

        ..............................................................................
        """,
        epilog="""
        ..............................................................................
        End help!  author:Vincent  2019-1-6 
        """)

    # required  参数是否为必须项[True|False]
    # nargs=     '+' 至少传入一个参数    '*' 传入0个或多个参数
    # default   默认值
    # help      帮助信息
    # type      参数属性默认都是字符串 [int|float|str]
    # choices   可选项 ['rock', 'paper', 'scissors']

    # 当前程序版本
    parser.add_argument('--version', action='version', version='%(prog)s 1.0', help='the version of this code')

    # tensorflow Session 运行参数设置
    parser.add_argument('--session__allow_soft_placement', action='store_false',
                        help='Allow device soft device placement [default True]')
    parser.add_argument('--session__log_device_placement', action='store_false',
                        help='Log placement of ops on devices [default True]')
    parser.add_argument('--session__allow_growth', action='store_false',
                        help='Allowing GPU memory growth [default True]')
    parser.add_argument('--session__per_process_gpu_memory_fraction', default=0.9, type=float,
                        help='only allocate 0.9 of the total memory of each GPU [default 0.9]')

    # 日志保存路径
    parser.add_argument('--log__path', default=None,
                        help='日志保存路径，默认为 None, 即当前目录下的 _log 文件夹.')
    # 日志文件名称
    parser.add_argument('--log__name', default=None,
                        help='日志文件名称，默认为 None，即以 {ProjectName}_{time}.log 为文件名.')
    # 控制台输出的日志级别默认为 DEBUG
    parser.add_argument('--log__level_s', default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='控制台输出的日志级别默认为 DEBUG  [DEBUG,INFO,WARNING,ERROR,CRITICAL]')
    # 日志文件输出的日志级别默认为 DEBUG
    parser.add_argument('--log__level_f', default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志文件输出的日志级别默认为 DEBUG  [DEBUG,INFO,WARNING,ERROR,CRITICAL]')

    # 训练样本 文件名称
    parser.add_argument('--trainData__path', required=False, help='训练原始数据文件')
    # 训练数据 TFRecord 临时存放路径
    parser.add_argument('--trainData__TFRecord', default=None,
                        help='训练数据转换成 TFRecord 存放的临时文件夹. 默认为 None，即当前目录下的 _TFRecordData 文件夹')
    # 训练数据 循环训练迭代次数
    parser.add_argument('--trainData__epochs', default=3, type=int,
                        help='训练数据 循环训练迭代次数 [default 3]')
    # 训练数据 每批次样本数量
    parser.add_argument('--trainData__batch_size', default=100, type=int,
                        help='训练数据 每批次样本数量 [default 100]')
    # 训练数据 随机乱序 buffer 大小
    parser.add_argument('--trainData__buffer_size', default=10000, type=int,
                        help='训练数据 随机乱序 buffer 大小, 若为 0 则不乱序 [default 1000]')

    # 运行模式
    parser.add_argument('--train__pattern', default='Check', choices=['Check', 'Train', 'Dev', 'Delete'],
                        help='运行模式，Check 查看模型结构.  [Check,Train,Dev,Delete]')
    # 模型名称
    parser.add_argument('--train__model_name', default='CNN', choices=['BP', 'CNN', 'CNN2'],
                        help='训练模型名称  [BP,CNN,CNN2]')
    # L1 正则权重系数
    parser.add_argument('--train__l1_scale', type=float, default=None,
                        help='L1 正则权重系数 默认为 None  disables the regularizer [dytpe: float]')
    # L2 正则权重系数
    parser.add_argument('--train__l2_scale', type=float, default=None,
                        help='L2 正则权重系数 默认为 None  disables the regularizer [dytpe: float]')
    # dropout 参数保留概率
    parser.add_argument('--train__dropout_keep_p', type=float, default=0.9,
                        help='dropout 参数保留概率 [dytpe: float, default 0.9  [0, 1]]')
    # 打印 训练过程打印频率
    parser.add_argument('--train__print_step', type=int, default=100,
                        help='每训练 N 步打印一次信息[dytpe: int, default 100]')
    # 保留最近的N个训练过程模型
    parser.add_argument('--train__cpkt_kept_num', type=int, default=1,
                        help='保留最近的N个训练过程模型.[dytpe: int, default 1]')
    # 模型训练优化器
    parser.add_argument('--train__optimizer',  default='Adam', choices=['Adam', 'SGD'],
                        help='模型训练优化器，默认为 Adam.[Adam,SGD]')
    # 学习率 learning_rate  Adam 优化器下，学习率无效
    parser.add_argument('--train__learning_rate', default='0.8,100,0.9',
                        help="模型训练学习率 '0.8,100,0.9' 初始学习率、衰减步数、衰减系数  或常数 '1.0'")

    args = parser.parse_args()

    # ##############################################################################################################
    # # TODO  本地调试使用， 实际使用时请注释
    # args.trainData__path = r'C:\Users\Vincent\Desktop\TensorFlow_Demo\Data\fashion'
    # # # args.log__path = r'D:\Temp\Temp\log'
    # # # args.log__name = 'test'
    # # args.session__log_device_placement = False
    # # args.trainData__epochs = 2
    # # args.trainData__batch_size = 200
    # # args.train__pattern = "Delete"
    # args.train__pattern = "Train"
    # # args.train__pattern = "Dev"
    # args.train__model_name = 'BP'
    # # args.train__optimizer = "SGD"
    # ##############################################################################################################

    # 开始时间 浮点型
    args.Start_time_float = time.time()
    # 开始时间 字符型
    args.Start_time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())

    # 当前目录路径
    args.CWD = os.path.abspath(os.path.dirname(__file__))

    # 当前项目名称
    args.ProjectName = os.path.basename(os.path.dirname(__file__))

    # 判断日志路径及名称，若为 None 则在当前路径下创建 _log 文件夹
    if args.log__path is None:
        log_path = os.path.join(args.CWD, '_log')
    else:
        assert os.path.isdir(os.path.dirname(args.log__path)), \
            "No Such Log Path:  {}   Please redetermine your log path! ".format(os.path.dirname(args.log__path))
        log_path = args.log__path

    # 判断是否存在日志文件夹， 若不存在则重新创建
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    # 日志文件名称, 若为 None 则以 {ProjectName}_{time}.log 为 日志文件名称
    if args.log__name is None:
        log_name = '{}_{}.log'.format(args.Start_time_str, args.train__model_name)
    else:
        log_name = '{}.log'.format(args.log__name)

    # 日志文件名称路径
    args.Logger_file = os.path.join(log_path, log_name)
    # 日志文件设置
    logger_set(path=args.Logger_file, name="TensorFlow", f_level=args.log__level_f, s_level=args.log__level_s)

    # 读取日志模块handler
    args.Logger = logging.getLogger("TensorFlow")

    # Tensorboard 结构参数保存路径
    args.Tensorboard_log_path = os.path.join(args.CWD, '_log', "TensorboardLog", '%s_%s' %
                                             (args.Start_time_str, args.train__model_name))

    #  若为 None 则默认存在当前目录下的 _TFRecordData 文件夹中
    if args.trainData__TFRecord is None:
        args.TFRecord_path = os.path.join(args.CWD, '_TFRecordData')
        # 判断是否存在 TFRecord 数据 文件夹 TFRecord ，若无则重新创建
        if not os.path.isdir(args.TFRecord_path):
            os.mkdir(args.TFRecord_path)
    else:
        # 不存在此路径，报错
        assert os.path.exists(args.trainData__TFRecord), \
            args.Logger.error("No Such TFRecord Path:  {}   Please redetermine your TFRecord path! ".
                              format(args.trainData__TFRecord))
        args.TFRecord_path = args.trainData__TFRecord

    # 判断是否存在临时文件夹 ，若无则重新创建
    if not os.path.isdir(os.path.join(args.CWD, '_temp')):
        os.mkdir(os.path.join(args.CWD, '_temp'))
    # 当前项目的临时文件夹
    if not os.path.isdir(os.path.join(args.CWD, '_temp', args.train__model_name)):
        os.mkdir(os.path.join(args.CWD, '_temp', args.train__model_name))
    # 训练过程 CPKT 存放路径
    args.Train_cpkt_path = os.path.join(args.CWD, '_temp', args.train__model_name,
                                        "{}.cpkt".format(args.Start_time_str))

    # 判断是否存在结果文件夹 ，若无则重新创建
    if not os.path.isdir(os.path.join(args.CWD, '_result')):
        os.mkdir(os.path.join(args.CWD, '_result'))

    # ############################# 打印参数 ################################################
    # 读取 args 中的所有参数
    args_info = args.__dict__
    args.Logger.info('{}{:20s}{}'.format('* '*35, "     Arguments    ", '* '*35))
    for i, args_i in enumerate(sorted(args_info)):
        args.Logger.info('args_{:02d} ---  {:s} : {}'.format(i+1, args_i, args_info[args_i]))
    args.Logger.info('{}'.format('* '*80))

    return args


def main():

    # 参数初始化
    args = argument_init()

    # ======================================================== 清空已有临时文件
    if args.train__pattern == "Delete":
        args.Logger.info("Delete temporary data {}.".format("_temp _result _TFRecord __pycache__"))
        import shutil
        if os.path.isdir(os.path.join(args.CWD, "_temp")):
            shutil.rmtree(os.path.join(args.CWD, "_temp"))
        if os.path.isdir(os.path.join(args.CWD, "_result")):
            shutil.rmtree(os.path.join(args.CWD, "_result"))
        if os.path.isdir(os.path.join(args.CWD, "_TFRecord")):
            shutil.rmtree(os.path.join(args.CWD, "_TFRecord"))
        if os.path.isdir(os.path.join(args.CWD, "__pycache__")):
            shutil.rmtree(os.path.join(args.CWD, "__pycache__"))

    # ======================================================== 模型查看、训练
    elif args.train__pattern == "Train" or args.train__pattern == "Check":
        # 读取训练数据 TFRecord 数据
        data_batch_ops, labels_batch_ops = get_train_tfrecord_data(args)

        # 创建神经网络模型
        if args.train__model_name == 'BP':
            from CODE import BP
            model = BP(data_batch_ops.shape[-1].value, labels_batch_ops.shape[-1].value,
                       l1_scale=args.train__l1_scale, l2_scale=args.train__l2_scale)
        elif args.train__model_name == 'CNN':
            from CODE import CNN
            model = CNN(data_batch_ops.shape[-1].value, labels_batch_ops.shape[-1].value,
                        l1_scale=args.train__l1_scale, l2_scale=args.train__l2_scale)
        elif args.train__model_name == 'CNN2':
            from CODE import CNN2
            model = CNN2(data_batch_ops.shape[-1].value, labels_batch_ops.shape[-1].value,
                         l1_scale=args.train__l1_scale, l2_scale=args.train__l2_scale)
        else:
            args.Logger.error('Error train model name [%s],Please enter the correct model. ' % args.train__model_name)
            raise Exception('Error train model name [%s],Please enter the correct model. ' % args.train__model_name)

        # 打印模型信息
        model.model_information()

        # 查看模型结构
        if args.train__pattern == "Check":
            args.Logger.info("Check The Tensorflow Model [{}].".format(args.train__model_name))
            # 保存模型图结构  tensorboard --logdir=log_path
            with tf.Session() as sess:
                # 初始化所有变量
                sess.run(tf.global_variables_initializer())
                tf.summary.FileWriter(args.Tensorboard_log_path, sess.graph)

        # 模型训练
        elif args.train__pattern == "Train":
            from CODE import run_trainer

            run_trainer(args, model, data_batch_ops, labels_batch_ops)

    # ======================================================== 模型验证
    elif args.train__pattern == "Dev":

        from sklearn.metrics import classification_report, confusion_matrix
        from CODE import get_dev_data

        # 获取验证数据
        x_data_dev, y_data_dev = get_dev_data(args)

        # 循环验证已训练好各个模型
        for m, model_name in enumerate(sorted(os.listdir(os.path.join(args.CWD, '_result')))):
            model_path = os.path.join(args.CWD, '_result', model_name)
            args.Logger.info("{:02d}th Model  {}  verification.".format(m+1, model_path))

            # 创建新的计算图
            with tf.Graph().as_default():

                # 读取已训练好的  .pb 模型文件
                with open(model_path, "rb") as f:
                    output_graph_def = tf.GraphDef()
                    output_graph_def.ParseFromString(f.read())
                    tf.import_graph_def(output_graph_def, name="")

                with tf.Session() as sess:
                    # 提取模型中的 输入数据、 预测值
                    dropout_keep_p = sess.graph.get_tensor_by_name("input/dropout_keep_p:0")
                    x_input = sess.graph.get_tensor_by_name("input/input_x:0")
                    y_predict = sess.graph.get_tensor_by_name("output/predict:0")
                    # y_predict_pro = sess.graph.get_tensor_by_name("output/predict_p:0")

                    # 利用模型预测
                    y_predict = sess.run(y_predict, feed_dict={x_input: x_data_dev, dropout_keep_p: 1})

                    # 分类结果报告
                    dev_report = classification_report(y_data_dev, y_predict, digits=3)
                    args.Logger.info('Evaluation Report: \n{}'.format(dev_report))
                    # 分类结果混淆矩阵
                    dev_confusion_matrix = confusion_matrix(y_data_dev, y_predict)
                    args.Logger.info('Evaluation Confusion_matrix: \n{}\n\n'.format(dev_confusion_matrix))


if __name__ == '__main__':

    main()
