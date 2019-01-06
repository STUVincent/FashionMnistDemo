此程序以 FashionMnist 作为数据集，用 TensorFlow 实现的分类模型 Demo

python 3.5  TensorFlow 1.12.0
直接运行 demo.py 脚本， 或者通过linux_train.sh 脚本

以下为运行后的文件列表
C:\Users\Vincent\Desktop\FashionMnist_Demo  Folder Size:【121.191 MB】
                          
└-- C:\Users\Vincent\Desktop\FashionMnist_Demo
    ├-- .gitignore    【0.12 KB   2019-01-06 16:38:25】
    ├-- demo.py    【15.47 KB   2019-01-06 18:23:44】
    └-- linux_train.sh    【1.00 KB   2019-01-06 18:25:46】
    └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git
        ├-- config    【0.13 KB   2019-01-06 16:37:47】
        ├-- description    【0.07 KB   2019-01-06 16:37:47】
        ├-- HEAD    【0.02 KB   2019-01-06 16:37:47】
        └-- index    【0.90 KB   2019-01-06 16:51:14】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\hooks
            ├-- applypatch-msg.sample    【0.47 KB   2019-01-06 16:37:47】
            ├-- commit-msg.sample    【0.88 KB   2019-01-06 16:37:47】
            ├-- fsmonitor-watchman.sample    【3.25 KB   2019-01-06 16:37:47】
            ├-- post-update.sample    【0.18 KB   2019-01-06 16:37:47】
            ├-- pre-applypatch.sample    【0.41 KB   2019-01-06 16:37:47】
            ├-- pre-commit.sample    【1.60 KB   2019-01-06 16:37:47】
            ├-- pre-push.sample    【1.32 KB   2019-01-06 16:37:47】
            ├-- pre-rebase.sample    【4.78 KB   2019-01-06 16:37:47】
            ├-- pre-receive.sample    【0.53 KB   2019-01-06 16:37:47】
            ├-- prepare-commit-msg.sample    【1.46 KB   2019-01-06 16:37:47】
            └-- update.sample    【3.53 KB   2019-01-06 16:37:47】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\info
            └-- exclude    【0.23 KB   2019-01-06 16:37:47】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects\24
                ├-- 397b5e1acfd8b7b0a15a71cd3f81b7720084ce    【0.40 KB   2019-01-06 16:44:45】
                └-- 91ea078e91dcd9d3ca225e16d74ea4f72f60d2    【0.40 KB   2019-01-06 16:40:44】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects\29
                └-- ce3691591d6bdd5a5461ed87b3eb0289edbeb2    【0.40 KB   2019-01-06 16:51:14】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects\72
                └-- 0ad819f63a0b47e4cb120e7e7a04a289fb4b0d    【0.40 KB   2019-01-06 16:47:24】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects\7c
                └-- 770a1eb6384590b4c9122500a53976901dabe8    【0.40 KB   2019-01-06 16:48:18】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects\f5
                └-- 0e1e25b92a513eb950e485afbbfef31bb153a1    【0.40 KB   2019-01-06 16:43:00】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects\f8
                └-- 918143ab5bc8832cf230ce228888ef6f1a1f36    【0.40 KB   2019-01-06 16:42:04】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects\info
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\objects\pack
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\refs
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\refs\heads
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.git\refs\tags
    └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\.idea
        ├-- FashionMnist_Demo.iml    【0.45 KB   2019-01-06 16:41:01】
        ├-- misc.xml    【0.22 KB   2019-01-06 16:39:07】
        ├-- modules.xml    【0.28 KB   2019-01-06 16:39:07】
        ├-- vcs.xml    【0.18 KB   2019-01-06 16:41:01】
        └-- workspace.xml    【17.90 KB   2019-01-06 18:24:30】
    └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\CODE
        ├-- LOG.py    【3.86 KB   2019-01-06 16:41:39】
        └-- __init__.py    【0.51 KB   2019-01-06 16:51:44】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\CODE\model
            ├-- basic_layer.py    【9.54 KB   2019-01-06 18:10:08】
            ├-- model_common.py    【11.02 KB   2019-01-06 18:17:49】
            ├-- nn_model.py    【27.10 KB   2019-01-06 18:17:16】
            └-- __init__.py    【0.33 KB   2019-01-06 16:49:55】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\CODE\model\__pycache__
                ├-- basic_layer.cpython-35.pyc    【7.02 KB   2019-01-06 18:11:25】
                ├-- model_common.cpython-35.pyc    【6.84 KB   2019-01-06 18:17:59】
                ├-- nn_model.cpython-35.pyc    【12.31 KB   2019-01-06 18:17:59】
                └-- __init__.cpython-35.pyc    【0.51 KB   2019-01-06 16:51:49】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\CODE\train_data
            ├-- load_train_data.py    【4.77 KB   2019-01-06 16:48:28】
            ├-- tf_record.py    【3.38 KB   2019-01-06 17:58:43】
            └-- __init__.py    【0.46 KB   2019-01-06 16:49:34】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\CODE\train_data\__pycache__
                ├-- load_train_data.cpython-35.pyc    【3.54 KB   2019-01-06 16:51:49】
                ├-- tf_record.cpython-35.pyc    【2.30 KB   2019-01-06 18:05:31】
                └-- __init__.cpython-35.pyc    【0.59 KB   2019-01-06 16:51:49】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\CODE\__pycache__
            ├-- LOG.cpython-35.pyc    【2.50 KB   2019-01-06 16:51:49】
            └-- __init__.cpython-35.pyc    【0.55 KB   2019-01-06 16:51:49】
    └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\Data
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\Data\fashion
            ├-- t10k-images-idx3-ubyte.gz    【4318.46 KB   2019-01-04 17:06:45】
            ├-- t10k-labels-idx1-ubyte.gz    【5.03 KB   2019-01-04 17:06:45】
            ├-- train-images-idx3-ubyte.gz    【25802.62 KB   2019-01-04 17:06:45】
            └-- train-labels-idx1-ubyte.gz    【28.82 KB   2019-01-04 17:06:45】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\Data\MNIST_data
            ├-- t10k-images-idx3-ubyte.gz    【1610.23 KB   2017-04-15 17:39:00】
            ├-- t10k-labels-idx1-ubyte.gz    【4.44 KB   2017-04-15 17:39:01】
            ├-- train-images-idx3-ubyte.gz    【9680.10 KB   2017-04-15 17:38:39】
            └-- train-labels-idx1-ubyte.gz    【28.20 KB   2017-04-15 17:38:41】
    └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_log
        ├-- 20190106102624_BP.log    【9.70 KB   2019-01-06 18:27:16】
        ├-- 20190106102719_CNN.log    【11.04 KB   2019-01-06 18:28:29】
        └-- 20190106102832_CNN.log    【7.08 KB   2019-01-06 18:28:35】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_log\TensorboardLog
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_log\TensorboardLog\20190106102624_BP
                └-- events.out.tfevents.1546770385.3faed569eecb    【744.45 KB   2019-01-06 18:27:17】
            └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_log\TensorboardLog\20190106102719_CNN
                └-- events.out.tfevents.1546770439.3faed569eecb    【3382.24 KB   2019-01-06 18:28:30】
    └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_result
        ├-- 20190106102624_BP.pb    【3469.04 KB   2019-01-06 18:27:16】
        └-- 20190106102719_CNN.pb    【40.78 KB   2019-01-06 18:28:29】
    └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_temp
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_temp\BP
            ├-- 20190106102624.cpkt-600.data-00000-of-00001    【10384.11 KB   2019-01-06 18:27:16】
            ├-- 20190106102624.cpkt-600.index    【0.77 KB   2019-01-06 18:27:16】
            ├-- 20190106102624.cpkt-600.meta    【113.90 KB   2019-01-06 18:27:16】
            └-- checkpoint    【0.14 KB   2019-01-06 18:27:16】
        └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_temp\CNN
            ├-- 20190106102719.cpkt-600.data-00000-of-00001    【98.52 KB   2019-01-06 18:28:29】
            ├-- 20190106102719.cpkt-600.index    【0.94 KB   2019-01-06 18:28:29】
            ├-- 20190106102719.cpkt-600.meta    【120.14 KB   2019-01-06 18:28:29】
            └-- checkpoint    【0.14 KB   2019-01-06 18:28:29】
    └-- C:\Users\Vincent\Desktop\FashionMnist_Demo\_TFRecordData
        ├-- fashion_info.pickle    【0.36 KB   2019-01-06 16:51:50】
        └-- fashion_train.tfrecord    【64083.50 KB   2019-01-06 16:52:13】
