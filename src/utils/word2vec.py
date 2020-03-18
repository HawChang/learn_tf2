#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 2:33 PM
# @Author : ZhangHao
# @File   : word2vec.py.py
# @Desc   : 


import gensim
import numpy as np


def train_word2vec(file_path, vec_path, previous_vec_path=None, size=100, window=5, min_count=2, workers=4, epochs=5):
    """
    训练词向量
    :param file_path: str，语料库地址，每行一句话，每句话已切分，空格分隔
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
    if previous_vec_path is None:
        # 训练模型
        model = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers)
    else:
        model = gensim.models.Word2Vec.load(previous_vec_path)
        #model.train(sentences, total_examples=len(list(sentences)), epochs=epochs)
        model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
        
    # 存储模型
    model.save(vec_path)


def load_word2vec(vec_path):
    
    model = gensim.models.Word2Vec.load(vec_path)
    # 获取token列表
    vocab_list = [word for word, Vocab in model.wv.vocab.items()]  # 存储 所有的 词语
    # 初始化一系列字典
    # token_id字典
    word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    # token_vec字典
    #word_vector = {}  # 初始化`[word : vector]`字典
    # 词向量矩阵
    # emb_matrix[token_id] = token_vec
    # emb_matrix[0]为全零词向量，用于padding
    # 因此矩阵shape=(vocab_size+1, vec_size)
    emb_matrix = np.zeros((len(vocab_list) + 1, model.vector_size))
    
    # 更新一系列字典
    for i in range(len(vocab_list)):
        # 当前token
        word = vocab_list[i]
        # 更新token_id字典
        word_index[word] = i + 1
        # 更新token_vec字典
        #word_vector[word] = model.wv[word]
        # 更新词向量矩阵
        emb_matrix[i + 1] = model.wv[word]
    
    return word_index, emb_matrix


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
    seg_data_path = './output/data.txt'
    vec_path = './output/word2vec_128d'
    pre_vec_path = './output/word2vec_128d'
    
    train_word2vec(
        file_path=seg_data_path,
        vec_path=vec_path,
        previous_vec_path=pre_vec_path,
        epochs=20)
    
    _test_word2vec(vec_path)
    pass
