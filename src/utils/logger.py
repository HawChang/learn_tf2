#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 3:49 PM
# @Author : ZhangHao
# @File   : logger.py
# @Desc   : 


import logging

"""
Logger的级别：
1. DEBUG
2. INFO
3. WARNING
4. ERROR
5. CRITICAL
"""


class Logger(object):
	_is_init = False
	
	def __init__(self):
		if not self._is_init:
			logging.basicConfig(
				# filename="log/run.log",
				level=logging.DEBUG,
				format="[%(asctime)s][%(filename)s:%(funcName)s:%(lineno)s][%(levelname)s]:%(message)s",
				datefmt='%Y-%m-%d %H:%M:%S')
			# ch = logging.StreamHandler()
			self.logger = logging.getLogger()
			# self.logger.addHandler(ch)
			self._is_init = True
	
	def get_logger(self):
		return self.logger


if __name__ == "__main__":
	pass