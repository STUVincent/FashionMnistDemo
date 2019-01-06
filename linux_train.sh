#!/bin/bash
# 重启服务命令, 启动日志存放到 _DockerLog 文件夹中
cd /OutFolder
mkdir _DockerLog
# 训练 1
python /OutFolder/demo.py  --trainData__path '/OutFolder/Data/fashion' --train__pattern "Train" --train__model_name 'BP' --trainData__epochs 2 --trainData__batch_size 200 > /OutFolder/_DockerLog/CMD_start_`date +%Y%m%d%H%M%S`.log
# 训练 2
python /OutFolder/demo.py  --trainData__path '/OutFolder/Data/fashion' --train__pattern "Train" --train__model_name 'CNN' --trainData__epochs 2 --trainData__batch_size 200 > /OutFolder/_DockerLog/CMD_start_`date +%Y%m%d%H%M%S`.log
# 验证训练模型
python /OutFolder/demo.py  --trainData__path '/OutFolder/Data/fashion' --train__pattern "Dev" > /OutFolder/_DockerLog/CMD_start_`date +%Y%m%d%H%M%S`.log
# 运行命令 将本地文件夹临时挂载到Docker里 （docker 退出后，文件仍存在 但不再挂载）
# docker run -itd -v C:\Users\Vincent\Desktop\FashionMnist_Demo:/OutFolder tensorflow/tensorflow:1.12.0-py3 /bin/bash /OutFolder/linux_train.sh
