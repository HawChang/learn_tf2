#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 3:45 PM
# @Author : ZhangHao
# @File   : jinyong_article_classification.py
# @Desc   : 分辨句子出自金庸的哪一本小说

import os
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/utils/" % _cur_dir)
sys.path.append("%s/model/" % _cur_dir)
import math

from utils.data_io import get_file_name_list, get_data, tokenizer, load_pkl, dump_pkl, write_to_file
from utils.seg import line_seg
from utils.logger import Logger
from utils.word2vec import train_word2vec, load_word2vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model.lstm_att import LstmAttModel

import numpy as np
import itertools
import config

log = Logger().get_logger()

# def load_train_test_data(data_path, label_path, token_id):
#     data_list = get_data(data_path)
#     label_list = get_data(label_path)
#     label_encoder = LabelEncoder()
#     label_list = label_encoder.fit_transform(label_list)
#     data_list = tokenizer(data_list, token_id)
#     train_x, test_x, train_y, test_y = train_test_split(data_list, label_list, test_size=0.15, shuffle=True)
#     return train_x, train_y, test_x, test_y, label_encoder
#
#
# def train(train_data, train_label, vec_path):
#     print("load word2vec...")
#     token_id, emb_mat = load_word2vec(vec_path)
#     print("emb matrix shape: %s" % str(emb_mat.shape))
#     # 加载训练数据
#     print("split train test")
#     train_x, train_y, test_x, test_y, label_encoder = load_train_test_data(train_data, train_label, token_id)
#
#     class_num = len(label_encoder.classes_)
#     print("class num = %d" % class_num)
#     vocab_size = len(token_id.keys())
#     print("vocab size = %d" % vocab_size)
#
#     print("train_y shape: %s" % str(train_y.shape))
#     print("test_y shape: %s" % str(test_y.shape))
#     print("train_y:")
#     print(train_y)
#     print("test_y:")
#     print(test_y)
#
#     batch_size = 128
#
#     train_ds = tf.data.Dataset.from_generator(
#             generator=lambda: iter(zip(train_x, train_y)),
#             output_types=(tf.int32, tf.int32),
#             output_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
#     train_ds = train_ds.shuffle(10000).padded_batch(batch_size, padded_shapes=([None], [])).prefetch(tf.data.experimental.AUTOTUNE)  # .shuffle(10000).batch(32)
#     test_ds = tf.data.Dataset.from_generator(
#             generator=lambda: iter(zip(test_x, test_y)),
#             #generator=lambda: gen(),
#             output_types=(tf.int32, tf.int32),
#             output_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
#     test_ds = test_ds.shuffle(10000).padded_batch(batch_size, padded_shapes=([None], [])).prefetch(tf.data.experimental.AUTOTUNE)  # shuffle(10000).batch(32)
#
#     print("train example:")
#     print(list(train_ds.take(3).as_numpy_iterator()))
#     print("creat train test dataset")
#
#     train_batch_num = math.ceil(train_y.shape[0] / float(batch_size))
#     test_batch_num = math.ceil(test_y.shape[0] / float(batch_size))
#     print("train batch num = %d" % train_batch_num)
#     print("test batch num = %d" % test_batch_num)
#
#     EPOCHS = 5
#
#     model = LstmAttModel(
#             class_num= class_num,
#             vocab_size= vocab_size,
#             emb_size= 128,
#             emb_matrix=emb_mat,
#             hidden_num=128)
#
#     optimizer = tf.keras.optimizers.Adam()
#     checkpoint = tf.train.Checkpoint(lstm_att_model=model, lstm_att_optimizer=optimizer)
#     manager = tf.train.CheckpointManager(checkpoint, directory='./model', checkpoint_name='model.ckpt', max_to_keep=5)
#     checkpoint.restore(tf.train.latest_checkpoint('./model'))
#     #checkpoint.restore('./model/model.ckpt-3')
#
#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#     test_loss = tf.keras.metrics.Mean(name='test_loss')
#     test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#     @tf.function(input_signature=(
#         tf.TensorSpec(shape=[None, None], dtype=tf.int32),
#         tf.TensorSpec(shape=[None], dtype=tf.int32),))
#     def train_step(batch_seq, batch_label):
#         with tf.GradientTape() as tape:
#             # training=True is only needed if there are layers with different
#             # behavior during training versus inference (e.g. Dropout).
#             predictions, att_weights = model(batch_seq, training=True)
#             loss = loss_object(batch_label, predictions)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         train_loss.update_state(loss)
#         train_accuracy.update_state(batch_label, predictions)
#
#     @tf.function(input_signature=(
#         tf.TensorSpec(shape=[None, None], dtype=tf.int32),
#         tf.TensorSpec(shape=[None], dtype=tf.int32),))
#     def test_step(batch_seq, batch_label):
#         # training=False is only needed if there are layers with different
#         # behavior during training versus inference (e.g. Dropout).
#         predictions, att_weights = model(batch_seq, training=False)
#         t_loss = loss_object(batch_label, predictions)
#
#         test_loss.update_state(t_loss)
#         test_accuracy.update_state(batch_label, predictions)
#
#     max_val_acc = 0
#
#     for epoch in range(EPOCHS):
#         print("=" * 50)
#         train_loss.reset_states()
#         train_accuracy.reset_states()
#         test_loss.reset_states()
#         test_accuracy.reset_states()
#
#         cur_batch = 0
#
#         for images, labels in train_ds:
#             # print("data shape : %s, label shape: %s" % (str(images.shape), str(labels.shape)))
#             cur_batch += 1
#             train_step(images, labels)
#             print('Epoch {}, train batch({}/{}): loss {:.6f}, acc: {:.2f}%'.format(
#                 epoch + 1,
#                 cur_batch,
#                 train_batch_num,
#                 train_loss.result(),
#                 train_accuracy.result() * 100))
#
#         for test_images, test_labels in test_ds:
#             test_step(test_images, test_labels)
#
#         test_los = test_loss.result()
#         test_acc = test_accuracy.result()
#
#         print('Epoch {}, val loss {:.6f}, acc: {:.2f}%'.format(
#             epoch + 1,
#             test_los,
#             test_acc * 100))
#
#         if max_val_acc < test_acc or True:
#             max_val_acc = test_acc
#             model_save_path = manager.save(checkpoint_number=epoch)
#             print("achieve best val acc, save model to %s" % model_save_path)
#         print("=" * 50)
#
#
# def check_att(train_data, train_label, vec_path):
#     print("load word2vec...")
#     token_id, emb_mat = load_word2vec(vec_path)
#     id_token = {v:k for k,v in token_id.items()}
#     print("emb matrix shape: %s" % str(emb_mat.shape))
#     # 加载训练数据
#     print("split train test")
#     train_x, train_y, test_x, test_y, label_encoder = load_train_test_data(train_data, train_label, token_id)
#
#     class_num = len(label_encoder.classes_)
#     print("class num = %d" % class_num)
#     vocab_size = len(token_id.keys())
#     print("vocab size = %d" % vocab_size)
#
#     print("train_y shape: %s" % str(train_y.shape))
#     print("test_y shape: %s" % str(test_y.shape))
#     print("train_y:")
#     print(train_y)
#     print("test_y:")
#     print(test_y)
#
#     batch_size = 128
#
#     test_ds = tf.data.Dataset.from_generator(
#             generator=lambda: iter(zip(test_x, test_y)),
#             #generator=lambda: gen(),
#             output_types=(tf.int32, tf.int32),
#             output_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
#     test_ds = test_ds.shuffle(10000).padded_batch(batch_size, padded_shapes=([None], [])).prefetch(tf.data.experimental.AUTOTUNE)  # shuffle(10000).batch(32)
#
#     print("test example:")
#     print(list(test_ds.take(3).as_numpy_iterator()))
#     print("creat train test dataset")
#
#     train_batch_num = math.ceil(train_y.shape[0] / float(batch_size))
#     test_batch_num = math.ceil(test_y.shape[0] / float(batch_size))
#     print("train batch num = %d" % train_batch_num)
#     print("test batch num = %d" % test_batch_num)
#
#     model = LstmAttModel(
#             class_num= class_num,
#             vocab_size= vocab_size,
#             emb_size= 128,
#             emb_matrix=emb_mat,
#             hidden_num=128)
#
#     optimizer = tf.keras.optimizers.Adam()
#     checkpoint = tf.train.Checkpoint(lstm_att_model=model, lstm_att_optimizer=optimizer)
#     manager = tf.train.CheckpointManager(checkpoint, directory='./model', checkpoint_name='model.ckpt', max_to_keep=5)
#     checkpoint.restore(tf.train.latest_checkpoint('./model')).expect_partial()
#
#     test_loss = tf.keras.metrics.Mean(name='test_loss')
#     test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#     @tf.function(input_signature=(
#         tf.TensorSpec(shape=[None, None], dtype=tf.int32),
#         tf.TensorSpec(shape=[None], dtype=tf.int32),))
#     def test_step(batch_seq, batch_label):
#         # training=False is only needed if there are layers with different
#         # behavior during training versus inference (e.g. Dropout).
#         predictions, att_weights = model(batch_seq, training=False)
#         t_loss = loss_object(batch_label, predictions)
#
#         test_loss.update_state(t_loss)
#         test_accuracy.update_state(batch_label, predictions)
#         return predictions, att_weights
#
#     max_val_acc = 0
#
#     test_loss.reset_states()
#     test_accuracy.reset_states()
#
#     cur_batch = 0
#
#     def display_check_att(test_datas, att_weights, predictions):
#         predictions = label_encoder.inverse_transform(np.argmax(predictions.numpy(), axis=1))
#         test_datas = test_datas.numpy()
#         print("test_datas shape = {}".format(test_datas.shape))
#         att_weights = np.squeeze(att_weights.numpy(), axis=-1)
#         print("att_weights shape = {}".format(att_weights.shape))
#         for data, att_weight, prediction in zip(test_datas, att_weights, predictions):
#             print("prediction: {}".format(prediction))
#             #print("prediction: " % label_encoder.inverse_transform([np.argmax(prediction.numpy())])[0])
#             for token_id, cur_weight in zip(data, att_weight):
#                 print("{}: {:.6f}".format(
#                     id_token[token_id],
#                     cur_weight,
#                     ))
#
#     for test_datas, test_labels in test_ds.take(1):
#         predictions, att_weights = test_step(test_datas, test_labels)
#         display_check_att(test_datas, att_weights, predictions)
#         #break
#
#     test_los = test_loss.result()
#     test_acc = test_accuracy.result()
#
#     print('check loss {:.6f}, acc: {:.2f}%'.format(
#         test_los,
#         test_acc * 100))
#
#
# def train_vec():
#     sentences = get_data(
#             config.preprocessed_data_dir,
#             read_func=lambda x: x.split('\t')[1].split(' '))
#
#     train_word2vec(
#             sentences=sentences,
#             vec_path=config.word2vec_path,
#             previous_vec_path=config.word2vec_path,
#             size=config.emb_size,
#     )
#
# def load_train_test_data(data_path, label_path, token_id):
#     data_list = get_data(data_path)
#     label_list = get_data(label_path)
#     label_encoder = LabelEncoder()
#     label_list = label_encoder.fit_transform(label_list)
#     data_list = tokenizer(data_list, token_id)
#     train_x, test_x, train_y, test_y = train_test_split(data_list, label_list, test_size=0.15, shuffle=True)
#     return train_x, train_y, test_x, test_y, label_encoder
#
#
# def tokenizer(data_list, token_id):
#     token_ids = list()
#     for data in data_list:
#         token_id_seq = list()
#         for token in data.split(' '):
#             token_id_seq.append(token_id.get(token, 0))  # 把句子中的 词语转化为index
#         token_ids.append(token_id_seq)
#     return token_ids


class JinYongArticleClassifier(object):
    def __init__(self, reload=True):
        # 加载已有模型
        if reload:
            log.info("load word2vec...")
            self.token_id, self.id_token, self.emb_mat = load_word2vec(config.word2vec_path)
            log.info("emb matrix shape: {}".format(self.emb_mat.shape))
            self.vocab_size = len(self.token_id.keys())
            log.info("vocab size = {}".format(self.vocab_size))
            log.info("load label_encode...")
            self.label_encoder = load_pkl(config.label_encoder_path)
            #dump_pkl(self.label_encoder, config.label_encoder_path, overwrite=True)
            self.class_num = len(self.label_encoder.classes_)
            log.info("class num = {}".format(self.class_num))
            log.info("classes: {}".format(self.label_encoder.classes_))

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    def gen_data(self, data_path, label_encoder=None, encoding='utf-8'):
        """
        将指定文件中的数据和标签拿出
        :param data_path: str，文件地址
        :param encoding: str，文件编码
        :param label_encoder: LabelEncoder, 转换标签为id，如果为None则不转换
        :return: 数据二元组(token id序列, 标签字面)
        """
        
        def line_process(line):
            line = line.strip('\n')
            if len(line) == 0:
                return None
            label, data = line.split('\t')
            token_seq = [self.token_id.get(token, 0) for token in data.split(' ')]
            return token_seq, label
        
        data_list = get_data(data_path, read_func=line_process, encoding=encoding)
        
        if label_encoder is not None:
            data, label = zip(*data_list)
            label = label_encoder.transform(label)
            data_list = list(zip(data, label))
        return data_list
    
    def load_model(self, expect_partial=False):
        
        self.model = LstmAttModel(
            class_num=self.class_num,
            vocab_size=self.vocab_size,
            emb_size=config.emb_size,
            emb_matrix=self.emb_mat,
            hidden_num=config.lstm_size)
        self.optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(lstm_att_model=self.model, lstm_att_optimizer=self.optimizer)
        res = checkpoint.restore(tf.train.latest_checkpoint(config.model_dir))
        if expect_partial:
            res.expect_partial()
        
        self.manager = tf.train.CheckpointManager(
            checkpoint,
            directory=config.model_dir,
            checkpoint_name=config.ckpt_prefix,
            max_to_keep=config.max_to_keep)

    @tf.function(input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),))
    def train_step(self, batch_seq, batch_label):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions, att_weights = self.model(batch_seq, training=True)
            loss = self.loss_object(batch_label, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(batch_label, predictions)

    @tf.function(input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),))
    def test_step(self, batch_seq, batch_label):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, att_weights = self.model(batch_seq, training=False)
        t_loss = self.loss_object(batch_label, predictions)
        self.val_loss.update_state(t_loss)
        self.val_accuracy.update_state(batch_label, predictions)
        return predictions, att_weights

    def train(self, train_data_path, val_data_path):
        # 获取训练数据
        train_data = self.gen_data(train_data_path, label_encoder=self.label_encoder)
        # 如果没有类别转换 则需要新生成一个
        if self.label_encoder is None:
            log.info("label_encoder is None, generate one")
            train_x, train_y = zip(*train_data)
            self.label_encoder = LabelEncoder()
            train_y = self.label_encoder.fit_transform(train_y)
            train_data = zip(train_x, train_y)
            log.info('dump label_encoder.pkl')
            dump_pkl(self.label_encoder, config.label_encoder_path)
            self.class_num = len(self.label_encoder.classes_)
            log.info("class num = {}".format(self.class_num))
        val_data = self.gen_data(val_data_path, label_encoder=self.label_encoder)
        
        train_ds = tf.data.Dataset.from_generator(
                generator=lambda: iter(train_data),
                output_types=(tf.int32, tf.int32),
                output_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
        train_ds = train_ds.shuffle(10000).padded_batch(config.batch_size, padded_shapes=([None], []))\
                .prefetch(tf.data.experimental.AUTOTUNE)  # .shuffle(10000).batch(32)
        
        val_ds = tf.data.Dataset.from_generator(
                generator=lambda: iter(val_data),
                output_types=(tf.int32, tf.int32),
                output_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
        val_ds = val_ds.shuffle(10000).padded_batch(config.batch_size, padded_shapes=([None], []))\
                .prefetch(tf.data.experimental.AUTOTUNE)  # shuffle(10000).batch(32)
    
        log.info("train example:")
        log.info(list(train_ds.take(1).as_numpy_iterator()))
        log.info("creat train test dataset")
        
        train_batch_num = math.ceil(len(train_data) / float(config.batch_size))
        val_batch_num = math.ceil(len(val_data) / float(config.batch_size))
        log.info("train batch num = %d" % train_batch_num)
        log.info("val batch num = %d" % val_batch_num)

        self.load_model()
        for epoch in range(config.epochs):
            print("=" * 50)
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
        
            for cur_batch, (batch_seq, batch_label) in enumerate(train_ds):
                # print("data shape : %s, label shape: %s" % (str(images.shape), str(labels.shape)))
                self.train_step(batch_seq, batch_label)
                print('Epoch {}, train batch({}/{}): loss {:.6f}, acc: {:.2f}%'.format(
                    epoch + 1,
                    cur_batch + 1,
                    train_batch_num,
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100))
        
            for batch_seq, batch_label in val_ds:
                self.test_step(batch_seq, batch_label)
        
            total_val_loss = self.val_loss.result()
            total_val_acc = self.val_accuracy.result()
        
            print('Epoch {}, val loss {:.6f}, acc: {:.2f}%'.format(
                epoch + 1,
                total_val_loss,
                total_val_acc * 100))
            
            model_save_path = self.manager.save(checkpoint_number=epoch)
            log.info("save model to %s" % model_save_path)
            print("=" * 50)

    def display_check_att(self, test_datas, att_weights, predictions):
        predictions = self.label_encoder.inverse_transform(np.argmax(predictions.numpy(), axis=1))
        test_datas = test_datas.numpy()
        log.info("test_datas shape = {}".format(test_datas.shape))
        att_weights = np.squeeze(att_weights.numpy(), axis=-1)
        log.info("att_weights shape = {}".format(att_weights.shape))
        for data, att_weight, prediction in zip(test_datas, att_weights, predictions):
            print("prediction: {}".format(prediction))
            for token_id, cur_weight in zip(data, att_weight):
                if token_id == 0:
                    break
                print("{}: {:.6f}".format(
                        self.id_token[token_id],
                        cur_weight))

    def check_att(self, val_data_path):
        val_data = self.gen_data(val_data_path, label_encoder=self.label_encoder)
        
        val_ds = tf.data.Dataset.from_generator(
            generator=lambda: iter(val_data),
            output_types=(tf.int32, tf.int32),
            output_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
        val_ds = val_ds.shuffle(10000).padded_batch(config.batch_size, padded_shapes=([None], [])) \
            .prefetch(tf.data.experimental.AUTOTUNE)  # shuffle(10000).batch(32)
        
        self.load_model(expect_partial=True)
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()
        for batch_seq, batch_label in val_ds:
            self.test_step(batch_seq, batch_label)
            predictions, att_weights = self.test_step(batch_seq, batch_label)
            self.display_check_att(batch_seq, att_weights, predictions)
            break
        total_val_loss = self.val_loss.result()
        total_val_acc = self.val_accuracy.result()

        print('val loss {:.6f}, acc: {:.2f}%'.format(
            total_val_loss,
            total_val_acc * 100))


def split_train_test():
    data = get_data(config.preprocessed_data_dir)
    train_data, val_data = train_test_split(data, test_size=config.test_ratio, shuffle=config.shuffle)
    write_to_file(train_data, config.train_data_path)
    write_to_file(val_data, config.val_data_path)

def main():
    #from preprocess import preprocess
    #preprocess(
    #    data_dir=config.origin_data_dir,
    #    output_dir=config.preprocessed_data_dir,
    #)
    #split_train_test()
    classifier = JinYongArticleClassifier(reload=True)
    #classifier.train(
    #    train_data_path=config.train_data_path,
    #    val_data_path=config.val_data_path,
    #)
    classifier.check_att(
        val_data_path=config.val_data_path,
    )

if __name__ == "__main__":
    main()
