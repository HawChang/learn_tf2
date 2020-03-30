#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 2:33 PM
# @Author : ZhangHao
# @File   : word2vec.py.py
# @Desc   : 


import os

import gensim
import numpy as np

from data_io import get_data


def train_word2vec_by_dir(data_dir, vec_path, read_func=lambda x: x, previous_vec_path=None,
                          size=100, window=5, min_count=2, workers=4, epochs=5):
    """
    根据指定的数据集地址和文本读取函数
    :param data_dir:
    :param vec_path: str，词向量存储地址
    :param read_func:
    :param previous_vec_path: str, 之前的词向量模型地址, 如果有，则加载该模型继续训练
    :param size: int，词向量维度
    :param window: int，窗口大小
    :param min_count: int，词频率阈值
    :param workers: int，并行数
    :param epochs: int, 训练轮数
    :return: None
    """
    sentences = get_data(data_dir, read_func=read_func)
    # 根据数据训练词向量
    train_word2vec(sentences, vec_path, previous_vec_path, size, window, min_count, workers, epochs)


def train_word2vec_by_file(file_path, vec_path, previous_vec_path=None, size=100, window=5, min_count=2, workers=4, epochs=5):
    """
    根据指定语料库文件训练词向量
    :param file_path: str，语料库文件地址，其中每行一句话 每句话中各token由空白格分隔
    :param vec_path: str，词向量存储地址
    :param previous_vec_path: str, 之前的词向量模型地址, 如果有，则加载该模型继续训练
    :param size: int，词向量维度
    :param window: int，窗口大小
    :param min_count: int，词频率阈值
    :param workers: int，并行数
    :param epochs: int, 训练轮数
    :return: None
    """
    # 读取分词后的文本
    sentences = gensim.models.word2vec.LineSentence(file_path)
    # 根据数据训练词向量
    train_word2vec(sentences, vec_path, previous_vec_path, size, window, min_count, workers, epochs)


def train_word2vec(sentences, vec_path, previous_vec_path=None, size=100, window=5, min_count=2, workers=4, epochs=5):
    """
    根据给出的数据训练词向量
    :param sentences: list[list[str]]，二维数组形式的语料库，每一句话为一个一维数组，若干句组成一个二维数组
    :param vec_path: str，词向量存储地址
    :param previous_vec_path: str, 之前的词向量模型地址, 如果有，则加载该模型继续训练
    :param size: int，词向量维度
    :param window: int，窗口大小
    :param min_count: int，词频率阈值
    :param workers: int，并行数
    :param epochs: int, 训练轮数
    :return: None
    """
    if previous_vec_path is None or not os.path.exists(previous_vec_path):
        # 训练模型
        model = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers)
    else:
        model = gensim.models.Word2Vec.load(previous_vec_path)
        model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
        
    # 存储模型
    model.save(vec_path)


class TokenEncoder(object):
    def __init__(self, token_id_dict, oov='<unk>'):
        """
        初始化token和id的映射
        :param token_id_dict:
        :param oov: 未登录词, 这里只是指明oov是哪一个, 需要在训练词向量时有该oov,
        """
        self.token_id = token_id_dict
        assert ' ' not in self.token_id, "token ' ' should not be used, it's reserved for padding"
        assert 0 not in self.token_id.values(), "token_id 0 should not be used, it's reserved for padding"
        self.token_id[' '] = 0
        self.oov_id = self.token_id[' ']
        if oov is not None:
            assert oov in self.token_id, "oov('{}') not in token, it should exist when training word2vec".format(oov)
            self.oov_id = self.token_id[oov]
        self.id_token = {v: k for k, v in self.token_id.items()}
        self.vocab_size = len(self.token_id)

    def transform(self, token):
        return self.token_id[token] if token in self.token_id else self.oov_id
    
    def inverse_transform(self, token_id):
        return self.id_token[token_id]


def load_word2vec(vec_path, oov='<unk>'):

    model = gensim.models.Word2Vec.load(vec_path)
    # 获取token列表
    token_list = [word for word, _ in model.wv.vocab.items()]  # 存储 所有的 词语
    # 初始化一系列字典

    # token_id字典
    token_index = dict()  # 初始化 `{token : index}` ，后期 tokenize 语料库就是用该词典。

    # 词向量矩阵
    # emb_matrix[token_id] = token_vec
    # emb_matrix[0]为全零词向量，用于padding
    # 因此矩阵shape=(vocab_size+1, vec_size)
    emb_matrix = np.zeros((len(token_list) + 1, model.vector_size))

    # 更新一系列字典
    for index, token in enumerate(token_list):
        # 更新token_id字典
        token_index[token] = index + 1
        # 更新词向量矩阵
        emb_matrix[index + 1] = model.wv[token]

    token_encoder = TokenEncoder(token_index, oov)

    return token_encoder, emb_matrix


def _test_word2vec(vec_path):
    """
    测试一下
    :param vec_path: 词向量文件地址
    :return:
    """
    # 读取自己的词向量，并简单测试一下 效果。
    model = gensim.models.Word2Vec.load(vec_path)

    print('\n'.join(list(model.wv.vocab.keys())[:100]))
    
    def get_most_simliar(token, top_k=5):
        print('打印与\'%s\'最相近的%d个词语：' % (token, top_k), model.wv.most_similar(token, topn=top_k))
    
    get_most_simliar('郭靖')
    get_most_simliar('令狐冲')
    get_most_simliar('九阴真经')


if __name__ == '__main__':
    import config

    train_word2vec_by_dir(
        data_dir=config.preprocessed_data_dir,
        vec_path=config.word2vec_path,
        previous_vec_path=config.pre_word2vec_path,
        epochs=config.word2vec_epochs)

    _test_word2vec(config.word2vec_path)
