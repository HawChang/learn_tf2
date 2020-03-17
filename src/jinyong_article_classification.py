#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 3:45 PM
# @Author : ZhangHao
# @File   : jinyong_article_classification.py
# @Desc   : 分辨句子出自金庸的哪一本小说

import os
import sys

from utils.data_io import get_file_name_list, get_data, tokenizer
from utils.seg import line_seg
from utils.logger import Logger
from utils.word2vec import train_word2vec, load_word2vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model.lstm_att import LstmAttModel

log = Logger().get_logger()


def load_file_conf(conf_path):
	file_conf = dict()
	with open(conf_path, 'r', encoding='utf-8') as rf:
		for line in rf:
			parts = line.strip('\n').split('\t')
			assert len(parts) == 4, "conf part num != 4, actual = %d" % len(parts)
			file_conf[parts[0]] = (parts[1], int(parts[2]), int(parts[3]))
	return file_conf


def process(data_dir, output_data_path, output_label_path):
	file_conf = load_file_conf("./conf/article_id.conf")
	file_path_list = get_file_name_list(data_dir)
	data_list = list()
	with open(output_data_path, 'w', encoding='utf-8') as w_data, \
			open(output_label_path, 'w', encoding='utf-8') as w_label:
		for file_path in file_path_list:
			file_name = file_path[file_path.rindex('/')+1:]
			label, start_index, end_index = file_conf[file_name]
			log.info("process file : %s" % label)
			with open(file_path, "r", encoding='utf-8') as rf:
				for index, line in enumerate(rf):
					if index + 1 < start_index:
						continue
					if index >= end_index:
						break
					if len(line) == 0:
						continue
					w_data.write(' '.join(line_seg(line.strip('\n'))) + '\n')
					w_label.write(label + '\n')


def load_train_test_data(data_path, label_path, token_id):
	data_list = get_data(data_path)
	label_list = get_data(label_path)
	label_encoder = LabelEncoder()
	label_list = label_encoder.fit_transform(label_list)
	data_list = tokenizer(data_list, token_id)
	train_x, test_x, train_y, test_y = train_test_split(data_list, label_list)
	return train_x, train_y, test_x, test_y, label_encoder


def train(train_data, train_label, vec_path):
	print("load word2vec...")
	token_id, emb_mat = load_word2vec(vec_path)
	print("emb matrix shape: %s" % str(emb_mat.shape))
	# 加载训练数据
	print("split train test")
	train_x, train_y, test_x, test_y, label_encoder = load_train_test_data(train_data, train_label, token_id)
	
	class_num = len(label_encoder.classes_)
	print("class num = %d" % class_num)
	vocab_size = len(token_id.keys())
	print("vocab size = %d" % vocab_size)
	
	train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, padding="post")
	test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, padding="post")
	
	#train_y = tf.keras.utils.to_categorical(train_y, num_classes=class_num)
	#test_y = tf.keras.utils.to_categorical(test_y, num_classes=class_num)
	
	print("train_y shape: %s" % str(train_y.shape))
	print("test_y shape: %s" % str(test_y.shape))
	
	train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).take(128).batch(32)#.shuffle(10000).batch(32)
	test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).take(128).batch(32)#shuffle(10000).batch(32)
	print("creat train test dataset")
	
	EPOCHS = 5
	
	model = LstmAttModel(
			class_num= class_num,
			vocab_size= vocab_size,
			emb_size= 128,
			seq_length=None,
			batch_size=None,
			emb_matrix=emb_mat,
			hidden_num=256)
	
	#print(model.summary())
	
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
	test_loss = tf.keras.metrics.Mean(name='test_loss')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
	
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	
	optimizer = tf.keras.optimizers.Adam()
	
	@tf.function
	def train_step(batch_seq, batch_label):
		with tf.GradientTape() as tape:
			# training=True is only needed if there are layers with different
			# behavior during training versus inference (e.g. Dropout).
			predictions = model(batch_seq, training=True)
			loss = loss_object(batch_label, predictions)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		
		train_loss(loss)
		train_accuracy(batch_label, predictions)
	
	@tf.function
	def test_step(batch_seq, batch_label):
		# training=False is only needed if there are layers with different
		# behavior during training versus inference (e.g. Dropout).
		predictions = model(batch_seq, training=False)
		t_loss = loss_object(batch_label, predictions)
		
		test_loss(t_loss)
		test_accuracy(labels, predictions)
	
	for epoch in range(EPOCHS):
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()
		
		for images, labels in train_ds:
			# print("data shape : %s, label shape: %s" % (str(images.shape), str(labels.shape)))
			# print('Batch , Loss: {}, Accura')
			train_step(images, labels)
		
		for test_images, test_labels in test_ds:
			test_step(test_images, test_labels)
		
		template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
		print(template.format(epoch + 1,
							  train_loss.result(),
							  train_accuracy.result() * 100,
							  test_loss.result(),
							  test_accuracy.result() * 100))


def main():
	data_dir = "./data/jinyong/"
	seg_data_path = './output/data.txt'
	label_path = './output/label.txt'
	vec_path = './output/word2vec_128d'
	
	## 预处理 生成切分好的数据和对应label文件
	#process(
	#	data_dir=data_dir,
	#	output_data_path=seg_data_path,
	#	output_label_path=label_path)
	#
	## 训练词向量
	#train_word2vec(
	#	file_path=seg_data_path,
	#	vec_path=vec_path,
	#	previous_vec_path=vec_path,
	#	size=128)
	
	train(
		train_data=seg_data_path,
		train_label=label_path,
		vec_path=vec_path)


if __name__ == "__main__":
	main()
	pass
