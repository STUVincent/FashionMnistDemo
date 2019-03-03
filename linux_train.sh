#!/bin/bash
# 启动服务命令, 启动日志存放到 _DockerLog 文件夹中
cd /OutFolder
mkdir _DockerLog
# 训练 1
python /OutFolder/main.py \
--trainData__path /OutFolder/Data/fashion \
--train__pattern Train \
--model__name model.cnn_v2 \
--train__optimizer SGD \
--trainData__batch_size 1000 \
--train__learning_rate 0.1 \
--train__learning_rate_decay_steps 1000 \
--train__learning_rate_decay_rate 0.98 \
--train__dropout_keep_p 0.8 \
--train__print_step 10 \
--trainData__epochs 10 \
--train__pretrained_model /OutFolder/_result/model.cnn_v2/20190302-015517/model-20190302-015517.ckpt-30000 \
> /OutFolder/_DockerLog/CMD_start_`date +%Y%m%d%H%M%S`.log
# 验证训练模型
python /OutFolder/main.py  --trainData__path /OutFolder/Data/fashion --model__name model.bp --train__pattern "Dev" > /OutFolder/_DockerLog/CMD_start_`date +%Y%m%d%H%M%S`.log
# 运行命令 将本地文件夹临时挂载到Docker里 （docker 退出后，文件仍存在 但不再挂载）
# sudo nvidia-docker run -it -v /home/vincent/Desktop/FashionMnist_Demo:/OutFolder tensorflow/tensorflow:1.12.0-gpu-py3 /bin/bash /OutFolder/linux_train.sh
# Tensorboard 查看训练过程 
# sudo docker run -it -p 6006:6006 -v /home/vincent/Desktop/FashionMnist_Demo/_log/TensorboardLog:/TensorboardLog tensorflow/tensorflow:1.12.0-py3 tensorboard --logdir /TensorboardLog