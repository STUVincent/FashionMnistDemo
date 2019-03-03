# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
初始化
# 当前项目: FashionMnist_Demo
# 创建时间: 2019/3/1 16:59 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import os
import time
import argparse
import logging
from .LOG import logger_set                    # 获取日志设置函数
from .Configure import GlobalConstants   # 导入全局常量


# 输入参数设置、全局参数、日志初始化
def argument_init(argv):
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
    parser.add_argument('--session__log_device_placement', action='store_true',
                        help='Log placement of ops on devices [default False]')
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
    parser.add_argument('--trainData__path', required=True, help='训练原始数据文件')
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
    # 调用模型名称
    parser.add_argument('--model__name', required=True, type=str, default=None,
                        help='调用模型名称 model.bp')
    # 运行模式
    parser.add_argument('--train__pattern', default='Check', choices=['Check', 'Train', 'Dev', 'Delete'],
                        help='运行模式，Check 查看模型结构.  [Check,Train,Dev,Delete]')
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
    parser.add_argument('--train__cpkt_kept_num', type=int, default=3,
                        help='保留最近的N个训练过程模型.[dytpe: int, default 1]')
    # 每训练 N 步保存一次 训练过程模型参数 CheckPoint
    parser.add_argument('--train__checkpoint_step', type=int, default=100,
                        help='保留最近的N个训练过程模型.[dytpe: int, default 100]')
    # 模型训练优化器
    parser.add_argument('--train__optimizer', choices=['SGD', 'ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='模型训练优化器，默认为 ADAM.[Adam,SGD]',  default='ADAM')
    # 初始学习率
    parser.add_argument('--train__learning_rate', type=float,
                        help='Initial learning rate.', default=0.1)
    # 学习率步长
    parser.add_argument('--train__learning_rate_decay_steps', type=int, default=100,
                        help='Number of epochs between learning rate decay.')
    # 学习率衰减系数
    parser.add_argument('--train__learning_rate_decay_rate', type=float, default=0.9,
                        help='Learning rate decay factor.')
    # 预训练模型，默认为None，从零开始训练
    parser.add_argument('--train__pretrained_model', type=str, default=None,
                        help='Load a pretrained model before training starts. "model-20170512-110547.ckpt-250000"')

    args = parser.parse_args(argv)

    # ##############################################################################################################
    # 开始时间 浮点型
    args.Start_time_float = time.time()
    # 开始时间 字符型
    # 时区差  当前时间为北京 东八区时间  此处操作为解决Docker中以 零时区 为标准而导致的时间混乱
    time_zone = 8 - int(time.strftime('%z', time.localtime()))/100
    args.Start_time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()+time_zone*60*60))

    # 项目目录路径  【当前文件在CODE 中，其上层目录】
    args.CWD = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
        log_name = '{}_{}.log'.format(args.Start_time_str, args.model__name)
    else:
        log_name = '{}.log'.format(args.log__name)

    # 日志文件名称路径
    args.Logger_file = os.path.join(log_path, log_name)
    # 日志文件设置
    logger_set(path=args.Logger_file, name="TF", f_level=args.log__level_f, s_level=args.log__level_s)

    # 读取日志模块handler
    args.Logger = logging.getLogger("TF")

    # Tensorboard 结构参数保存路径
    args.Tensorboard_log_path = os.path.join(args.CWD, '_log', "TensorboardLog", '%s_%s' %
                                             (args.Start_time_str, args.model__name))

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
    if not os.path.isdir(os.path.join(args.CWD, '_result')):
        os.mkdir(os.path.join(args.CWD, '_result'))
    # 当前项目的临时文件夹
    if not os.path.isdir(os.path.join(args.CWD, '_result', args.model__name)):
        os.mkdir(os.path.join(args.CWD, '_result', args.model__name))
    # 训练过程 CPKT 存放路径
    args.Train_cpkt_path = os.path.join(args.CWD, '_result', args.model__name, args.Start_time_str)

    # 判断是否存在结果文件夹 ，若无则重新创建
    if not os.path.isdir(os.path.join(args.CWD, '_temp')):
        os.mkdir(os.path.join(args.CWD, '_temp'))

    # ############################# 打印用户输入超参数 #####################################
    args.Logger.info('{}{:20s}{}'.format('* '*25, "     User Input   ", '* '*25))
    user_input = ""
    for argv_i in argv:
        if argv_i.startswith("-"):
            if len(user_input.strip()) > 0:
                args.Logger.info("Input: {}".format(user_input.strip()))
            user_input = ""
        user_input = user_input + "   " + argv_i
    args.Logger.info("Input: {}".format(user_input.strip()))

    # ############################# 打印参数 ################################################
    # 读取 args 中的所有参数
    args_info = args.__dict__
    args.Logger.info('{}{:20s}{}'.format('* '*25, "     Arguments    ", '* '*25))
    for i, args_i in enumerate(sorted(args_info)):
        args.Logger.info('args_{:02d} ---  {:s} : {}'.format(i+1, args_i, args_info[args_i]))

    # ############################# 打印全局常量 ############################################
    # 读取 GlobalConstants 中的所有全局常量
    constant_info = [constant_i for constant_i in GlobalConstants.__dict__ if not constant_i.startswith("__")]
    args.Logger.info('{}{:20s}{}'.format('* '*25, "  GlobalConstants ", '* '*25))
    for i, constant_i in enumerate(sorted(constant_info)):
        args.Logger.info('Constant_{:02d} ---  {:s} : {}'.format(i+1, constant_i, getattr(GlobalConstants, constant_i)))
    args.Logger.info('{}'.format('* '*60))

    return args
