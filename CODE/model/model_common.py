# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
优化器、 TensorBoard 设置、 模型训练等公共函数

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/1/6 16:43 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import os
import time
import tensorflow as tf


# 计算余弦距离
def tf_cos_dist(tf_np_a, tf_np_b):
    with tf.name_scope('CosDist'):
        # 计算 np_a 的 2范数
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(tf_np_a), axis=1))
        x2_norm = tf.sqrt(tf.reduce_sum(tf.square(tf_np_b), axis=1))

        # 分子
        numerator = tf.reduce_sum(tf.multiply(tf_np_a, tf_np_b), axis=1)
        # 分母  [防止分母为 0 输出为Nan 加上一个极小数 1e-8 ]
        denominator = tf.multiply(x1_norm, x2_norm) + 1e-8

        # 余弦距离
        cos_distance = numerator / denominator

    return cos_distance


# 计算类别的 F1 Score
def tf_f1_score(label_true, label_pred):
    """
    :param label_true:  实际标签值   np.array([1, 1, 2, 0, 1, 1]) 必须为 int 类型
    :param label_pred:  预测标签值   np.array([1, 1, 1, 0, 1, 1]) 必须为 int 类型
    :return:  类别的 Weight权重下 F1 Score [sklearn.metrics.classification_report ]
    """
    with tf.name_scope('F1_Score'):
        # 类别数量
        label_num = tf.cast(tf.reduce_max(label_true)+1, tf.int32)

        # 将实际值转换成 one_hot矩阵
        label_true_array = tf.one_hot(label_true, label_num, dtype=tf.int32)
        # 将预测值转换成 one_hot矩阵
        label_pred_array = tf.one_hot(label_pred, label_num, dtype=tf.int32)

        # 计算各真正例，真反例，假正例，假反例的数量
        tp = tf.count_nonzero(label_pred_array * label_true_array, axis=0)
        # tn = tf.count_nonzero((label_pred_array - 1) * (label_true_array-1), axis=0)  # 真反例
        fp = tf.count_nonzero(label_pred_array * (label_true_array - 1), axis=0)
        fn = tf.count_nonzero((label_pred_array - 1) * label_true_array, axis=0)

        # 转换为浮点型，方便计算
        tp = tf.cast(tp, tf.float64)
        fp = tf.cast(fp, tf.float64)
        fn = tf.cast(fn, tf.float64)

        # 准确率
        precision = tp / (tp + fp + 1e-8)
        # 召回率
        recall = tp / (tp + fn + 1e-8)
        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # 类别数量权重
        weights = tf.reduce_sum(label_true_array, axis=0)
        weights /= tf.reduce_sum(weights)

        # Weight 权重方式修正在总 F1 Score
        f1 = tf.reduce_sum(f1 * weights)

        return f1


# 优化器
def train_op(args, model):
    # 训练步长 全局变量。 加入 Model 模型中
    model.global_step = tf.Variable(0, name="global_step", trainable=False)

    # 优化器
    with tf.name_scope('train'):
        if args.train__optimizer == 'Adam':
            trainer = tf.train.AdamOptimizer(1e-3).minimize(model.loss, global_step=model.global_step)  # Adam

        elif args.train__optimizer == 'SGD':
            # 学习率参数设置
            learning_rate_p = [float(p_i) for p_i in args.train__learning_rate.split(',')]

            if len(learning_rate_p) > 1:  # 渐变学习率
                learning_rate_base = learning_rate_p[0]  # 初始学习率
                learning_rate_steps = learning_rate_p[1]  # 衰减速度
                learning_rate_decay = learning_rate_p[2]  # 衰减系数

                # 指数下降学习率   初始学习率、初始步数 、衰减步数、衰减系数  staircase=True 阶梯状衰减学习率( 默认为 True)
                model.learning_rate = tf.train.exponential_decay(learning_rate_base, model.global_step,
                                                                 learning_rate_steps, learning_rate_decay,
                                                                 staircase=True)
            else:  # 固定参数学习率
                model.learning_rate = tf.constant(learning_rate_p[0], name='Constant_learning_rate')

            # SGD 使用梯度下降法   global_step=global_step 指数衰减学习率时设置
            trainer = tf.train.GradientDescentOptimizer(model.learning_rate).\
                minimize(model.loss, global_step=model.global_step)

        else:
            trainer = None
            args.Logger.error('Error train optimizer [{}]， please input  Adam | SGD '.format(args.train_op))
            raise Exception('Error train optimizer [%s]， please input  Adam | SGD ' % args.train_op)

    return trainer


# 变量监控程序[将模型中保存到 TensorBoard 变量域的变量 加入监测]
def variable_summaries(logger):
    with tf.name_scope('Monitor'):
        # Tensorboard 命名域中的监测变量
        for tensor in tf.get_collection('Tensorboard'):
            try:
                var = tf.get_default_graph().get_tensor_by_name(tensor.name)

                with tf.name_scope(var.op.name):
                    # 若张量为多维数据，则计算其 均值、方差、最大最小值及其数据分布
                    if var.shape[-1] > 1:
                        mean = tf.reduce_mean(var)
                        tf.summary.scalar('max', tf.reduce_max(var))
                        tf.summary.scalar('min', tf.reduce_min(var))
                        tf.summary.scalar('mean', mean)
                        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
                        tf.summary.histogram('histogram', var)
                    # 若张量只有一个值，则只数值分布
                    else:
                        tf.summary.histogram('histogram', var)
            except Exception as error:  # 变量不存在
                logger.warn('Warning!!!  {}'.format(error))


# Tensorboard 监测
def tensorboard_monitor(args, model):
    # 训练过程参数 变化跟踪[验证集准确率、损失值]
    with tf.name_scope('Evaluation'):
        tf.summary.scalar('accuracy', model.accuracy)
        tf.summary.scalar('loss', model.loss)

        # SGD 优化器下才将 学习率 存入 tensorboard 中
        if args.train__optimizer == 'SGD':
            tf.summary.scalar('learning_rate', model.learning_rate)

    # 变量监控程序[将保存到 TensorBoard 变量域的变量 加入监测]
    variable_summaries(args.Logger)
    # 合并所有的summary
    merged = tf.summary.merge_all()

    return merged


# 训练过程
def run_trainer(args, model, data_batch_ops, labels_batch_ops):

    # =============================  优化器
    trainer = train_op(args, model)

    # =============================  Tensorboard 监测
    merged = tensorboard_monitor(args, model)

    # =============================  开始训练
    # Session Config
    session_config = tf.ConfigProto(
        allow_soft_placement=args.session__allow_soft_placement,
        log_device_placement=args.session__log_device_placement)
    # Allowing GPU memory growth
    session_config.gpu_options.allow_growth = args.session__allow_growth  # 默认为 False
    # only allocate 80% of the total memory of each GPU
    session_config.gpu_options.per_process_gpu_memory_fraction = args.session__per_process_gpu_memory_fraction

    # 模型训练
    with tf.Session(config=session_config) as sess:
        args.Logger.info('{} {:^30s} {}'.format('-'*30, 'Start training', '-'*30))
        time_start = time.time()  # 开始训练时间

        saver = tf.train.Saver(max_to_keep=args.train__cpkt_kept_num)

        # 查看是否有训练过的 CPKT 文件
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(args.Train_cpkt_path))
        # 若本地已存在训练过CPKT文件，读取上次的训练参数，继续训练
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            args.Logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        # 若无训练过的CPKT文件，则初始化随机参数，开始训练
        else:
            args.Logger.info("Created model with fresh parameters.")
            # 初始化
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        tensorboard_writer = tf.summary.FileWriter(args.Tensorboard_log_path, sess.graph)

        try:
            # 循环提取数据，直到全部读取结束
            while True:
                # 提取 训练 Batch Data
                data_batch_i, labels_batch_i = sess.run([data_batch_ops, labels_batch_ops])

                # 训练数据
                feed_dict = {model.input_x: data_batch_i, model.input_y: labels_batch_i,
                             model.dropout_keep_p: args.train__dropout_keep_p}
                # 模型训练
                _, summary, step, loss, acc, f1_score = sess.run([trainer, merged, model.global_step, model.loss,
                                                                  model.accuracy, model.f1_score], feed_dict=feed_dict)
                # 记录训练集计算的参数
                tensorboard_writer.add_summary(summary, step)

                # 定期打印并保存训练过程
                if step % args.train__print_step == 0:
                    args.Logger.debug('{:6d}/{:d}  loss:{:.3f}  acc:{:.3f}  F1:{:.3f}'.format(step, args.TrainTime,
                                                                                              loss, acc, f1_score))

                    # 保存训练过程
                    saver.save(sess, args.Train_cpkt_path, global_step=step)

        except tf.errors.OutOfRangeError:
            # 训练结束
            args.Logger.info('QueueData read over! End of the training. Used Time {:.1f} Sec'.
                             format(time.time() - time_start))

        # ################################################################################################
        # 保存模型 [可用于迁移学习]
        from tensorflow.python.framework import graph_util
        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, [model.predict.op.name, model.predict_p.op.name])
        model_path_save = os.path.join(args.CWD, '_result', '{}_{}.pb'.format(args.Start_time_str,
                                                                              args.train__model_name))
        with tf.gfile.GFile(model_path_save, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        # 打印模型相关张量信息
        args.Logger.info('model saved path: {}'.format(model_path_save))
        args.Logger.info('dropout_keep_p  : {}'.format(model.dropout_keep_p))
        args.Logger.info('x_input     OP  : {}'.format(model.input_x))
        args.Logger.info('y_predict_p OP  : {}'.format(model.predict_p))
        args.Logger.info('y_predict   OP  : {}'.format(model.predict))
