#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/19 10:20 PM
# @Author : ZhangHao
# @File   : preprocess.py
# @Desc   : 将金庸的文件处理为格式规范的文件

import os
import config

from utils.data_io import get_file_name_list
from utils.logger import Logger
from utils.seg import line_seg

log = Logger().get_logger()


def preprocess(data_dir, output_dir):
    file_name_list = get_file_name_list(data_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for src_path in file_name_list:
        file_name = src_path[src_path.rindex('/') + 1:]
        label, start_index, end_index = config.origin_file_name_dict[file_name]
        dst_path = os.path.join(output_dir, file_name)
        log.info("process file: {} to {}".format(src_path, dst_path))
        with open(src_path, 'r', encoding='utf-8') as rf, \
                open(dst_path, 'w', encoding='utf-8') as wf:
            for index, line in enumerate(rf):
                line = line.replace('\t', ' ').strip()
                if index + 1 < start_index:
                    continue
                if index >= end_index:
                    break
                if len(line) == 0:
                    continue
                wf.write('{}\t{}\n'.format(label, ' '.join(line_seg(line))))


if __name__ == "__main__":
    preprocess(
        data_dir=config.origin_data_dir,
        output_dir=config.preprocessed_data_dir,
    )
