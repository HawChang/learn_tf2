#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 5:12 PM
# @Author : ZhangHao
# @File   : lstm_att.py
# @Desc   : lstm + att模型


import tensorflow as tf


class LstmAttModel(tf.keras.Model):
	def __init__(self, class_num, vocab_size, emb_size, seq_length, batch_size, emb_matrix=None, hidden_num=256):
		super().__init__()
		# 此处添加初始化代码（包含 call 方法中会用到的层），例如
		# layer1 = tf.keras.layers.BuiltInLayer(...)
		# layer2 = MyCustomLayer(...)
		self.class_num = class_num
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		#self.seq_length = seq_length
		#self.batch_size = batch_size
		if emb_matrix is None:
			self.emb = tf.keras.layers.Embedding(self.vocab_size, self.emb_size)  # [batch_size, seq_length, emb_size]
		else:
			self.emb = tf.keras.layers.Embedding(self.vocab_size, self.emb_size, weights=[emb_matrix], trainable=False) # [batch_size, seq_length, emb_size]
		self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_num))
		self.dense = tf.keras.layers.Dense(self.class_num, activation='relu')

	def call(self, inputs):
		x = self.emb(inputs)  # [batch_size, seq_length, emb_size]
		x = self.lstm(x)  # [batch_size, hidden_num * 2]
		x = self.dense(x)  # [batch_size, class_num]
		return x


def main():
	
	model = LstmAttModel(10, vocab_size=1000, emb_size=128, seq_length=100, batch_size=64)
	model.summary()


if __name__ == "__main__":
	main()