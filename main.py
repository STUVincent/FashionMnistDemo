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
# 版    本: V1.1
"""
import os
import time
import sys
import tensorflow as tf
from CODE import argument_init             # 参数初始化
from CODE import get_train_tfrecord_data   # 获取训练数据 TFRecord 形式
from CODE import GlobalConstants           # 导入全局常量
from CODE import print_model_info          # 打印模型训练变量、 损失函数、 TensorBoard 监控变量
from CODE import tensorboard_monitor       # Tensorboard 监测
from CODE import train_optimizer           # 优化器
from CODE import save_variables_and_metagraph  # 模型训练过程保存

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '  # 只输出error


# 模型训练主程序
def main(argv):

    # 参数初始化
    args = argument_init(argv)

    # ======================================================== 清空已有临时文件
    if args.train__pattern == "Delete":
        # 待删除文件夹
        del_fold = "_temp _result _TFRecord"
        args.Logger.info("Delete temporary data {}.".format(del_fold))
        import shutil
        for del_fold_i in del_fold.split():
            if os.path.isdir(os.path.join(args.CWD, del_fold_i)):
                shutil.rmtree(os.path.join(args.CWD, del_fold_i))

    # ======================================================== 模型查看、训练
    elif args.train__pattern == "Train" or args.train__pattern == "Check":

        import importlib
        # 模型导入
        try:
            network = importlib.import_module(args.model__name)
        except Exception as error:  # 导入失败，直接退出
            args.Logger.error("Import Module Failure:  {}".format(error))
            raise Exception("Import Module Failure:  {}".format(error))

        # 构造网络图
        model = network.inference(args.Logger, GlobalConstants.model_x_shape, GlobalConstants.model_y_shape,
                                  l1_scale=args.train__l1_scale, l2_scale=args.train__l2_scale)

        # 打印模型训练变量、 损失函数、 TensorBoard 监控变量
        print_model_info(args.Logger)

        # 查看模型结构
        if args.train__pattern == "Check":
            args.Logger.info("Check The Tensorflow Model [{}].".format(args.model__name))
            # 保存模型图结构  tensorboard --logdir=log_path
            with tf.Session() as sess:
                # 初始化所有变量
                sess.run(tf.global_variables_initializer())
                tf.summary.FileWriter(args.Tensorboard_log_path, sess.graph)

        # 模型训练
        elif args.train__pattern == "Train":

            # =============================  读取训练数据 TFRecord 数据
            data_batch_ops, labels_batch_ops = get_train_tfrecord_data(args)

            # =============================  优化器
            train_op = train_optimizer(args, model)

            # =============================  Tensorboard 监测
            merged = tensorboard_monitor(args.Logger, model['accuracy'], model['loss'], model['learning_rate'])

            # =============================  开始训练
            # Session Config
            session_config = tf.ConfigProto(
                allow_soft_placement=args.session__allow_soft_placement,
                log_device_placement=args.session__log_device_placement)
            # Allowing GPU memory growth
            session_config.gpu_options.allow_growth = args.session__allow_growth
            # only allocate 80% of the total memory of each GPU
            session_config.gpu_options.per_process_gpu_memory_fraction = args.session__per_process_gpu_memory_fraction

            # 模型训练
            with tf.Session(config=session_config).as_default() as sess:
                args.Logger.info('{} {:^30s} {}'.format('-'*30, 'Start training', '-'*30))

                # Initialize variables  初始化模型参数
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Tensorboard 记录对象
                tensorboard_writer = tf.summary.FileWriter(args.Tensorboard_log_path, sess.graph)

                # Create a saver创建一个saver用来保存或者从内存中回复一个模型参数
                saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.train__cpkt_kept_num)

                # 若本地已存在训练过CPKT文件，读取上次的训练参数，继续训练
                if args.train__pretrained_model:
                    args.Logger.info('Restoring pretrained model: %s' % args.train__pretrained_model)
                    saver.restore(sess, args.train__pretrained_model)
                # 若无训练过的CPKT文件，则从初始化随机参数开始训练
                else:
                    args.Logger.info("Created model with fresh parameters.")

                try:
                    # 循环提取 TFRecord 中数据，直到全部读取结束
                    while True:
                        # 提取 训练 Batch Data
                        data_batch_i, labels_batch_i = sess.run([data_batch_ops, labels_batch_ops])

                        # 训练数据
                        feed_dict = {model['input_x']: data_batch_i, model['input_y']: labels_batch_i,
                                     model['dropout_keep_p']: args.train__dropout_keep_p}
                        # 模型训练
                        _, step = sess.run([train_op, model['global_step']], feed_dict=feed_dict)

                        # 定期打印训练过程
                        if step % args.train__print_step == 0:
                            summary, loss, acc, f1_score = sess.run([merged, model['loss'], model['accuracy'],
                                                                     model['f1_score']], feed_dict=feed_dict)
                            # 记录训练集计算的参数
                            tensorboard_writer.add_summary(summary, step)
                            args.Logger.debug('{:6d}/{:d}  loss:{:.3f}  acc:{:.3f}  F1:{:.3f}'.format(
                                step, args.TrainTime, loss, acc, f1_score))

                        # 定期保存训练过程模型参数
                        if step % args.train__checkpoint_step == 0:
                            # 保存训练过程模型参数
                            args.Logger.debug('Save Checkpoint. ')
                            save_variables_and_metagraph(sess, saver, args.Train_cpkt_path, args.Start_time_str, step)

                except tf.errors.OutOfRangeError:
                    # 训练结束
                    args.Logger.info('QueueData read over!')
                # ################## 将参数写入TensorBoard 的 Text 中
                # TODO 打印需要关注的参数
                args_info = "【learning_rate】:{:.5f}  【dropout_keep_p】:{:.2f}  【epochs】:{:3d}  【batch_size】:{}" \
                            "【optimizer】:{} <br/>【pretrained_model】:{}".\
                    format(args.train__learning_rate, args.train__dropout_keep_p, args.trainData__epochs,
                           args.trainData__batch_size, args.train__optimizer, args.train__pretrained_model)

                argument_info = sess.run(tf.summary.text("ArgumentInfo",
                                                         tf.convert_to_tensor(args_info.replace(' ', '&nbsp;'))))
                tensorboard_writer.add_summary(argument_info)

                # 训练结束，打印训练总时长。
                args.Logger.info("FinishTraining! UsedTime:{:.1f} Sec.".format(time.time()-args.Start_time_float))
                # ################################################################################################
                # 保存 ***.pb 模型
                from tensorflow.python.framework import graph_util
                constant_graph = graph_util.convert_variables_to_constants(
                    sess, sess.graph_def, [model['predict'].op.name, model['predict_p'].op.name])
                model_path_save = os.path.join(args.CWD, '_result', args.model__name, args.Start_time_str,
                                               '{}_{}.pb'.format(args.Start_time_str, args.model__name))
                with tf.gfile.GFile(model_path_save, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

                # 打印模型相关张量信息
                args.Logger.info('model saved path: {}'.format(model_path_save))
                args.Logger.info('dropout_keep_p  : {}'.format(model['dropout_keep_p']))
                args.Logger.info('x_input     OP  : {}'.format(model['input_x']))
                args.Logger.info('y_predict_p OP  : {}'.format(model['predict_p']))
                args.Logger.info('y_predict   OP  : {}'.format(model['predict']))

    # ======================================================== 模型验证
    elif args.train__pattern == "Dev":

        from sklearn.metrics import classification_report, confusion_matrix
        from CODE import get_dev_data

        # 获取验证数据
        x_data_dev, y_data_dev = get_dev_data(args)

        model_num = 1  # 模型序号
        # 循环验证已训练好各个模型
        for model_name in sorted(os.listdir(os.path.join(args.CWD, '_result'))):
            model_folder = os.path.join(args.CWD, '_result', model_name)

            for train_time in sorted(os.listdir(model_folder)):
                model_path = os.path.join(model_folder, train_time)
                # ***.pb 模型名称
                model_pb_name = [file_i for file_i in os.listdir(model_path) if file_i.endswith('.pb')]

                if len(model_pb_name) > 0:
                    model_pb_path = os.path.join(model_path, model_pb_name[0])
                    args.Logger.info("{:02d}th Model  {}  verification.".format(model_num, model_pb_path))
                    model_num += 1  # 模型序号自加 1

                    # 创建新的计算图
                    with tf.Graph().as_default():

                        # 读取已训练好的  .pb 模型文件
                        with open(model_pb_path, "rb") as f:
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

    r"""  模型调试参数 
    --train__pattern Train 
    --train__print_step 5
    --trainData__path .\Data\fashion
    --train__optimizer SGD
    --model__name model.cnn_v2 
    --trainData__epochs 1
    """

    main(sys.argv[1:])
