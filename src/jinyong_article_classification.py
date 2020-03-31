#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 3:45 PM
# @Author : ZhangHao
# @File   : jinyong_article_classification.py
# @Desc   : 分辨句子出自金庸的哪一本小说

import math
import os
import sys

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/utils/" % _cur_dir)
sys.path.append("%s/model/" % _cur_dir)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config
from model.lstm_att import LstmAttModel
from utils.data_io import get_file_name_list, get_data, load_pkl, dump_pkl, write_to_file
from utils.logger import Logger
from utils.seg import line_seg
from utils.word2vec import train_word2vec_by_dir, load_word2vec

log = Logger().get_logger()


class JinYongArticleClassifier(object):
    def __init__(self,
                 preprocess=False,
                 split_train_test=False,
                 gen_word2vec=False,
                 gen_label_encode=False,
                 reload_model=False,
                 expect_partial=False):
        """
        模型初始化
        :param preprocess: true则对原始数据预处理
        :param split_train_test: true则划分训练验证集
        :param gen_word2vec: true则训练词向量
        :param reload_model: true则载入已有模型参数
        :param gen_label_encode: true则生成新的标签id映射
        :param expect_partial: true则确定模型参数并未全部用到
        """
        # 模型依赖word2vec和label_encoder

        if preprocess:
            JinYongArticleClassifier.preprocess(config.origin_data_dir, config.preprocessed_data_dir)

        if split_train_test:
            JinYongArticleClassifier.split_train_test()

        # 加载word2vec
        if gen_word2vec:
            log.info('gen word2vec...')
            train_word2vec_by_dir(config.preprocessed_data_dir,
                                  read_func=lambda x: x.split('\t')[1].split(' '),
                                  vec_path=config.word2vec_path,
                                  previous_vec_path=config.pre_word2vec_path,
                                  size=config.emb_size,
                                  epochs=config.word2vec_epochs)

        log.info("load word2vec...")
        self.token_encoder, self.emb_mat = load_word2vec(config.word2vec_path, config.oov)
        log.info("emb matrix shape: {}".format(self.emb_mat.shape))
        log.info("vocab size = {}".format(self.token_encoder.vocab_size))

        # 加载label_encoder
        if gen_label_encode:
            log.info('gen label_encoder...')
            self.gen_label_encoder()

        log.info("load label_encoder...")
        self.label_encoder = load_pkl(config.label_encoder_path)
        log.info("class num = {}".format(self.label_encoder.classes_))
        log.info("classes: {}".format(self.label_encoder.classes_))

        # 加载model
        self.model = LstmAttModel(
            class_num=len(self.label_encoder.classes_),
            vocab_size=self.token_encoder.vocab_size,
            emb_size=config.emb_size,
            emb_matrix=self.emb_mat,
            hidden_num=config.lstm_size)

        self.optimizer = tf.keras.optimizers.Adam()

        checkpoint = tf.train.Checkpoint(lstm_att_model=self.model, lstm_att_optimizer=self.optimizer)
        if reload_model:
            res = checkpoint.restore(tf.train.latest_checkpoint(config.model_dir))
            if expect_partial:
                res.expect_partial()

        self.manager = tf.train.CheckpointManager(
            checkpoint,
            directory=config.model_dir,
            checkpoint_name=config.ckpt_prefix,
            max_to_keep=config.max_to_keep)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # 监测训练、验证时的loss和acc
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @staticmethod
    def preprocess(data_dir, output_dir):
        """
        预处理原始数据
        将小说生成(标签, 段落)的数据对
        :param data_dir: 原始数据地址
        :param output_dir: 输出数据地址
        :return: None
        """
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

    @staticmethod
    def split_train_test():
        """
        划分训练集和验证集
        :return: None
        """
        data = get_data(config.preprocessed_data_dir)
        # 生成训练集测试集
        train_data, val_data = train_test_split(data, test_size=config.test_ratio, shuffle=config.shuffle)
        log.info("write train data to file: {}".format(config.train_data_path))
        write_to_file(train_data, config.train_data_path)
        log.info("write val data to file: {}".format(config.val_data_path))
        write_to_file(val_data, config.val_data_path)

    @staticmethod
    def gen_label_encoder():
        """
        生成标签映射
        :return: None
        """
        labels = get_data(config.train_data_path, read_func=lambda x:x.split('\t')[0])
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        dump_pkl(label_encoder, config.label_encoder_path, overwrite=True)

    @tf.function(input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),))
    def train_step(self, batch_seq, batch_label):
        """
        训练步骤
        :param batch_seq: 数据
        :param batch_label: 标签
        :return: predictions: 预测标签
                 att_weights: attention层的值
        """
        with tf.GradientTape() as tape:
            predictions, att_weights = self.model(batch_seq)
            loss = self.loss_object(batch_label, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(batch_label, predictions)
        return predictions, att_weights

    @tf.function(input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),))
    def val_step(self, batch_seq, batch_label):
        """
        验证步骤
        :param batch_seq: 数据
        :param batch_label: 标签
        :return: predictions: 预测标签
                 att_weights: attention层的值
        """
        predictions, att_weights = self.model(batch_seq)
        t_loss = self.loss_object(batch_label, predictions)
        self.val_loss.update_state(t_loss)
        self.val_accuracy.update_state(batch_label, predictions)
        return predictions, att_weights

    def gen_dataset(self, data_path, shuffle_size=10000, batch_size=128, encoding='utf-8'):
        """
        将指定文件中的数据和标签拿出
        文件每行由'\t'分隔为两列, (标签、切词后各token用空白格分隔的字符串)
        例如: "标签\t单词1 单词2 单词3 ..."
        :param data_path: str，文件地址
        :param shuffle_size: int, 打乱时的buffer大小
        :param batch_size: int, 数据批量大小
        :param encoding: str，文件编码
        :param label_encoder: LabelEncoder, 转换标签为id，如果为None则不转换
        :return: 数据二元组(token id序列, 标签字面)
        """

        def line_process(line):
            line = line.strip('\n')
            if len(line) == 0:
                return None
            label, data = line.split('\t')
            token_seq = [self.token_encoder.transform(token) for token in data.split(' ')]
            label = self.label_encoder.transform([label])[0]
            return token_seq, label

        data_list = get_data(data_path, read_func=line_process, encoding=encoding)
        batch_num = math.ceil(len(data_list) / float(batch_size))
        train_ds = tf.data.Dataset.from_generator(
                generator=lambda: iter(data_list),
                output_types=(tf.int32, tf.int32),
                output_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
        train_ds = train_ds.shuffle(shuffle_size).padded_batch(batch_size, padded_shapes=([None], []))\
            .prefetch(tf.data.experimental.AUTOTUNE)
        return train_ds, batch_num

    def train(self, train_data_path, val_data_path):
        """
        训练
        :param train_data_path: 训练数据地址
        :param val_data_path: 验证数据地址
        :return: None
        """
        train_ds, train_batch_num = self.gen_dataset(train_data_path, batch_size=config.batch_size)
        log.info("train batch num = %d" % train_batch_num)
        val_ds, val_batch_num = self.gen_dataset(val_data_path, batch_size=config.batch_size)
        log.info("val batch num = %d" % val_batch_num)

        log.info("train example:")
        log.info(list(train_ds.take(1).as_numpy_iterator()))
        log.info("creat train test dataset")

        for epoch in range(config.epochs):
            log.info("=" * 50)
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            for cur_batch, (batch_seq, batch_label) in enumerate(train_ds):
                self.train_step(batch_seq, batch_label)
                log.info('Epoch {}, train batch({}/{}): loss {:.6f}, acc: {:.2f}%'.format(
                    epoch + 1,
                    cur_batch + 1,
                    train_batch_num,
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100))

            for batch_seq, batch_label in val_ds:
                self.val_step(batch_seq, batch_label)

            total_val_loss = self.val_loss.result()
            total_val_acc = self.val_accuracy.result()

            log.info('Epoch {}, val loss {:.6f}, acc: {:.2f}%'.format(
                epoch + 1,
                total_val_loss,
                total_val_acc * 100))

            model_save_path = self.manager.save(checkpoint_number=epoch)
            log.info("save model to %s" % model_save_path)
            log.info("=" * 50)

    def display_check_att(self, val_data, att_weights, predictions):
        """
        展示验证数据、attention值、预测标签，观察attention的情况
        :param val_data: 验证数据
        :param att_weights: attention权重值
        :param predictions: 预测标签
        :return: None
        """
        predictions = self.label_encoder.inverse_transform(np.argmax(predictions.numpy(), axis=1))
        val_data = val_data.numpy()
        log.info("test_datas shape = {}".format(val_data.shape))
        att_weights = np.squeeze(att_weights.numpy(), axis=-1)
        log.info("att_weights shape = {}".format(att_weights.shape))
        for data, att_weight, prediction in zip(val_data, att_weights, predictions):
            print("prediction: {}".format(prediction))
            for token_id, cur_weight in zip(data, att_weight):
                if token_id == 0:
                    break
                print("{}: {:.6f}".format(
                        self.token_encoder.inverse_transform(token_id),
                        cur_weight))

    def check_att(self, val_data_path):
        """
        验证
        :param val_data_path: 验证数据集
        :return: None
        """
        val_ds, batch_num = self.gen_dataset(val_data_path, batch_size=config.batch_size)
        log.info("val batch num = %d" % batch_num)

        self.val_loss.reset_states()
        self.val_accuracy.reset_states()
        for batch_seq, batch_label in val_ds:
            self.val_step(batch_seq, batch_label)
            predictions, att_weights = self.val_step(batch_seq, batch_label)
            # 观察attention权值
            self.display_check_att(batch_seq, att_weights, predictions)
            break
        total_val_loss = self.val_loss.result()
        total_val_acc = self.val_accuracy.result()

        print('val loss {:.6f}, acc: {:.2f}%'.format(
            total_val_loss,
            total_val_acc * 100))


def main():

    classifier = JinYongArticleClassifier(
        preprocess=False,
        split_train_test=True,
        gen_word2vec=True,
        gen_label_encode=True,
        reload_model=False,
        expect_partial=True,
    )

    classifier.train(
        train_data_path=config.train_data_path,
        val_data_path=config.val_data_path,
    )

    classifier.check_att(
        val_data_path=config.val_data_path,
    )


if __name__ == "__main__":
    main()
