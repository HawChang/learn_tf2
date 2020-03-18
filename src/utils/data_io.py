#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 3:48 PM
# @Author : ZhangHao
# @File   : data_io.py
# @Desc   : 


import codecs
import os
import pickle
import time

from sklearn.datasets import dump_svmlight_file
import numpy as np

from logger import Logger

log = Logger().get_logger()


def get_data(data_path, read_func=lambda x:x, encoding="utf-8", verbose=False):
    """获取该文件(或目录下所有文件)的数据
    [in]  data_path : str, 数据集地址
          verbose   : bool, 是否展示处理信息
    [out] data : list[str], 该文件(或目录下所有文件)的数据
    """
    file_list = get_file_name_list(data_path, verbose)
    data = list()
    for file_path in file_list:
        data.extend(read_from_file(file_path, read_func))
    return data


def get_file_name_list(data_path, verbose=False):
    """生成构成数据集的文件列表
        如果数据集地址是文件，则返回列表中只有该文件地址
        如果数据集地址是目录，则返回列表中包括该目录下所有文件名称(忽略'.'开头的文件)
    [in]  data_path : str, 数据集地址
          verbose   : bool, 是否展示处理信息
    [out] file_list : list[str], 数据集文件名称列表
    """
    from collections import deque
    file_list = list()
    path_stack = deque()
    path_stack.append(data_path)
    while len(path_stack) != 0:
        cur_path = path_stack.pop()
        log.debug("check data path: %s." % cur_path)
        # 首先检查原始数据是文件还是文件夹
        if os.path.isdir(cur_path):
            #log.debug("data path is directory.")
            files = os.listdir(cur_path)
            # 遍历文件夹中的每一个文件
            for file_name in files:
                # 如果文件名以.开头 说明该文件隐藏 不是正常的数据文件
                if len(file_name) == 0 or file_name[0] == ".":
                    continue
                file_path = os.path.join(cur_path, file_name)
                path_stack.append(file_path)
        elif os.path.isfile(cur_path):
            #log.info("data path is file. add to list.")
            file_list.append(cur_path)
        else:
            raise TypeError("unknown type of data path : %s" % cur_path)

    log.info("file list top 20:")
    for index, file_name in enumerate(file_list[:20]):
        log.info("#%d: %s" % (index + 1, file_name))
    return file_list


def read_from_file(file_path, read_func=lambda x:x, encoding="utf-8"):
    """加载文件中的词
    [in] file_path: str, 文件地址
    [out] word_list: list[str], 单词列表
    """
    word_list = list()
    with codecs.open(file_path, "r", encoding) as rf:
        for line in rf:
            word_list.append(read_func(line.strip("\n")))
    return word_list


def write_to_file(text_list, dst_file_path, write_func=lambda x:x, encoding="utf-8"):
    """将文本列表存入目的文件地址
    [in]  text_list: list[str], 文本列表
          dst_file_path: str, 目的文件地址
    """
    with codecs.open(dst_file_path, "w", encoding) as wf:
        wf.write("\n".join([write_func(x) for x in text_list]))


def load_pkl(pkl_path):
    """加载对象
    [in]  pkl_path: str, 对象文件地址
    [out] obj: class, 对象
    """
    with open(pkl_path, 'rb') as rf:
        return pickle.load(rf)


def dump_pkl(obj, pkl_path, overwrite=False):
    """存储对象
    [in]  obj: class, 对象
          pkl_path: str, 对象文件地址
          overwrite: bool, 是否覆盖，False则当文件存在时不存储
    """
    if len(pkl_path) == 0 or pkl_path is None:
        log.warning("pkl_path(\"%s\") illegal." % pkl_path)
    elif os.path.exists(pkl_path) and not overwrite:
        log.warning("pkl_path(\"%s\") already exist and not over write." % pkl_path)
    else:
        with open(pkl_path, 'wb') as wf:
            pickle.dump(obj, wf)
        log.debug("save to \"%s\" succeed." % pkl_path)


def label_encoder_save_as_class_id(label_encoder, class_id_path, conf_thres = 0.5):
    """将LabelEncoder对象转为def-user中的class_id.txt格式的形式存入指定文件
    [in]  label_encoder: class, 对象
          class_id_path: str, 存储文件地址
          conf_thres: float, 类别的阈值 这里只能统一设置
    """
    class_id_list = ["%d\t%s\t%f" % (index, str(class_name), conf_thres) for \
            index, class_name in enumerate(label_encoder.classes_)]
    write_to_file(class_id_list, class_id_path)
    log.debug("trans label_encoder to \"%s\" succeed." % class_id_path)


def dump_libsvm_file(X, y, file_path, zero_based=False):
    """将数据集转为libsvm格式 liblinear、xgboost、lightgbm都可以接收该格式
    [in]  X: array-like、sparse matrix, 数据特征
          y: array-like、sparse matrix, 类别结果
          file_path: string、file-like in binary model, 文件地址，或者二进制形式打开的可写文件
          zero_based: bool, true则特征id从0开始 liblinear训练时要求特征id从1开始 因此一般需要为False
    """
    log.debug("trans libsvm format data to %s." % file_path)
    start_time = time.time()
    dump_svmlight_file(X, y, file_path, zero_based=zero_based)
    log.info("cost_time : %.4fs" % (time.time() - start_time))


def tokenizer(data_list, token_id):
    token_ids = list()
    for data in data_list:
        token_id_seq = list()
        for token in data.split(' '):
            token_id_seq.append(token_id.get(token, 0))  # 把句子中的 词语转化为index
        token_ids.append(token_id_seq)
    return token_ids
