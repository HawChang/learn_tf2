#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/19 10:44 PM
# @Author : ZhangHao
# @File   : config.py
# @Desc   : 配置参数

import os

# 源数据文件夹
origin_data_dir = './data/jinyong/'

# 源数据的配置
origin_file_name_dict={
    'baimaxiaoxifeng.txt': ('白马啸西风', 9, 733),
    'yuanyangdao.txt': ('鸳鸯刀', 9, 299),
    'shediaoyingxiongzhuang.txt': ('射雕英雄传', 31, 8801),
    'xiaoaojianghu.txt': ('笑傲江湖', 9, 9817),
    'yitiantulongji.txt': ('倚天屠龙记', 9, 9711),
    'xiaokexing.txt': ('侠客行', 9, 4018),
    'feihuwaizhuan.txt': ('飞狐外传', 9, 4616),
    'ludingji.txt': ('鹿鼎记', 9, 11715),
    'lianchengjue.txt': ('连城诀', 9, 2739),
    'tianlongbabu.txt': ('天龙八部', 30, 12678),
    'bixuejian.txt': ('碧血剑', 9, 5108),
    'xueshanfeihu.txt': ('雪山飞狐', 9, 1285),
    'shendiaoxialv.txt': ('神雕侠侣', 9, 8549),
    'shujianenchoulu.txt': ('书剑恩仇录', 9, 4305),
    'yuenvjian.txt': ('越女剑', 9, 238),
}

# 预处理后的文件夹
# 预处理后，各行数据格式为:书名\t该行句子切词结果(空白格分隔)
preprocessed_data_dir = './data/jinyong_processed/'

# 数据集分训练集和测试集
intermediate_data_dir = './data/'
train_data_path = os.path.join(intermediate_data_dir, 'train_data.txt')
val_data_path = os.path.join(intermediate_data_dir, 'val_data.txt')

emb_size = 128
# 词向量文件地址
model_dir = './model'
word2vec_path = os.path.join(model_dir, 'word2vec_{}d'.format(emb_size))

# 类别名称id转换信息
label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')


# 批处理大小
batch_size = 128
epochs = 5
lstm_size = 128

# 模型保存的参数
ckpt_prefix = 'model.ckpt'
max_to_keep = 5

# 训练集测试集划分
test_ratio = 0.15
shuffle = True
