# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
训练过程 公共程序

# 当前项目: FashionMnist_Demo
# 创建时间: 2019/3/1 16:17 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5
# 版    本: V1.0
"""
import os
import tensorflow as tf


# 打印模型训练变量、 损失函数、 TensorBoard 监控变量
def print_model_info(logger):
    # 打印所有待训练的参数
    logger.info('Trainable Variables:')
    for trainable_variables_i in tf.trainable_variables():
        logger.info(
            '     {:30s} {:15s} {:s}'.format(trainable_variables_i.name, str(trainable_variables_i.shape),
                                             trainable_variables_i.dtype.name))
    # 打印所有损失函数
    logger.info('Loss Function :')
    for losses_i in tf.get_collection('losses'):
        #  losses集合中的所有张量
        logger.info('     {:s}'.format(losses_i.name))

    # 打印所有监测变量
    logger.info('Monitor Variables:')
    # 将全局常量中的监控 Tensor 加入  Tensorboard 命名空间中
    from CODE.Configure import GlobalConstants
    for tensor_i in GlobalConstants.monitor_tensor:
        try:
            tf.add_to_collection('Tensorboard', tf.get_default_graph().get_tensor_by_name(tensor_i))
        except Exception as error:  # 变量不存在
            logger.warn('     Warning!!!  {}'.format(error))
    for monitor_i in tf.get_collection('Tensorboard'):
        #  losses集合中的所有张量
        logger.info('     {:30s} {:15s}'.format(monitor_i.name, str(monitor_i.shape)))

    logger.info('{}'.format('* '*60))


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
                        # tf.summary.scalar('max', tf.reduce_max(var))
                        # tf.summary.scalar('min', tf.reduce_min(var))
                        tf.summary.scalar('mean', mean)
                        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
                        tf.summary.histogram('histogram', var)
                    # 若张量只有一个值，则只数值分布
                    else:
                        tf.summary.histogram('histogram', var)
            except Exception as error:  # 变量不存在
                logger.warn('Warning!!!  {}'.format(error))


# Tensorboard 监测
def tensorboard_monitor(logger, accuracy, loss, learning_rate):
    # 训练过程参数 变化跟踪[验证集准确率、损失值]
    with tf.name_scope('Evaluation'):
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

    # # 保存图片
    # var = tf.get_default_graph().get_tensor_by_name("Layer_Output/weight:0")
    # var = tf.reshape(var, [-1, 20, 10, 1])
    # tf.summary.image("IMG", var, max_outputs=100)

    # 变量监控程序[将保存到 TensorBoard 变量域的变量 加入监测]
    variable_summaries(logger)
    # 合并所有的summary
    merged = tf.summary.merge_all()

    return merged


# 优化器 学习率及训练步长将在此处加入 model 字典中
def train_optimizer(args, model):
    # 训练步长 全局变量。 加入 Model 模型中
    model['global_step'] = tf.Variable(0, name="global_step", trainable=False)

    # 指数下降学习率   初始学习率、初始步数 、衰减步数、衰减系数  staircase=True 阶梯状衰减学习率( 默认为 True)
    model['learning_rate'] = tf.train.exponential_decay(args.train__learning_rate, model['global_step'],
                                                        args.train__learning_rate_decay_steps,
                                                        args.train__learning_rate_decay_rate, staircase=True)

    # 优化器
    with tf.name_scope('train'):

        if args.train__optimizer == 'SGD':
            trainer = tf.train.GradientDescentOptimizer(model['learning_rate']).minimize(
                model['loss'], global_step=model['global_step'])

        elif args.train__optimizer == 'ADAGRAD':
            trainer = tf.train.AdagradOptimizer(model['learning_rate']).minimize(
                model['loss'], global_step=model['global_step'])

        elif args.train__optimizer == 'ADAM':
            trainer = tf.train.AdamOptimizer(model['learning_rate'], beta1=0.9, beta2=0.999, epsilon=0.1).minimize(
                model['loss'], global_step=model['global_step'])

        elif args.train__optimizer == 'ADADELTA':
            trainer = tf.train.AdadeltaOptimizer(model['learning_rate'], rho=0.9, epsilon=1e-6).minimize(
                model['loss'], global_step=model['global_step'])

        elif args.train__optimizer == 'RMSPROP':
            trainer = tf.train.RMSPropOptimizer(model['learning_rate'], decay=0.9, momentum=0.9, epsilon=1.0).minimize(
                model['loss'], global_step=model['global_step'])

        elif args.train__optimizer == 'MOM':
            trainer = tf.train.MomentumOptimizer(model['learning_rate'], 0.9, use_nesterov=True).minimize(
                model['loss'], global_step=model['global_step'])

        else:
            trainer = None
            args.Logger.error('Error train optimizer [{}]， please input  Adam | SGD '.format(args.train_op))
            raise Exception('Error train optimizer [%s]， please input  Adam | SGD ' % args.train_op)

    return trainer


# 保存模型
def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
    # Save the model checkpoint
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    if not os.path.exists(metagraph_filename):
        saver.export_meta_graph(metagraph_filename)
