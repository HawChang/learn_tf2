#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 2:49 PM
# @Author : ZhangHao
# @File   : seg.py
# @Desc   : 切词工具

import jieba


def line_seg(text):
	"""
	:param text: 待切分字符串
	:return: 切分列表
	"""
	return jieba.lcut(text)


def file_seg(file_path, seg_path, encoding='utf-8'):
	with open(file_path, 'r', encoding=encoding) as rf, \
			open(seg_path, 'w', encoding=encoding) as wf:
		for line in rf:
			wf.write(' '.join(line_seg(line.strip('\n'))) + '\n')


if __name__ == "__main__":
	pass