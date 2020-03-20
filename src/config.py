#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/19 10:44 PM
# @Author : ZhangHao
# @File   : config.py
# @Desc   : 配置参数

import os


# ----------- 预处理阶段 ------------
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

# ----------- 训练验证集划分阶段 ------------
# 划分验证集占比
test_ratio = 0.15

# 划分时是否打乱顺序
shuffle = True

# 数据集分训练集和测试集
intermediate_data_dir = './data/'
train_data_path = os.path.join(intermediate_data_dir, 'train_data.txt')
val_data_path = os.path.join(intermediate_data_dir, 'val_data.txt')

# ----------- 词向量生成阶段 ------------
# 词向量维度
emb_size = 128

# 词向量训练轮数
word2vec_epochs = 10

# 未登录词
# 设置的未登录词需要在训练词向量的语料里存在
# 例如 oov='<unk>' 则训练的语料里应该有token '<unk>'
# None表示没有
oov = None

# 词向量文件地址
model_dir = './model'
word2vec_path = os.path.join(model_dir, 'word2vec_{}d'.format(emb_size))
pre_word2vec_path = word2vec_path

# ----------- 类别id映射阶段 ------------
# 类别名称id转换信息
label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

# ----------- 模型训练阶段 ------------

# 批处理大小
batch_size = 128

# 训练轮数
epochs = 5

# LSTM神经元数
lstm_size = 128

# 模型保存文件前缀
ckpt_prefix = 'model.ckpt'

# 保存模型数目上限
max_to_keep = 5
