#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/17 2:13 PM
# @Author : ZhangHao
# @File   : att_layers.py
# @Desc   : 


import tensorflow as tf


class SelfAttention(tf.keras.Model):
    def __init__(self, units=256):
        super(SelfAttention, self).__init__()
        self.Wq = tf.keras.layers.Dense(units)
        self.Wk = tf.keras.layers.Dense(units)
        self.Wv = tf.keras.layers.Dense(units)
        self.units = units
    
    def call(self, inputs):
        # query shape = [batch_size, seq_length, hidden_size]
        q = self.Wq(inputs)
        k = self.Wk(inputs)
        v = self.Wv(inputs)
        
        attention_weights = tf.nn.softmax(tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) / (self.units ** 0.5), axis=1)
        # print("attention shape = {}".format(attention_weights.shape))
        
        context_vector = tf.matmul(attention_weights, v)
        # print("context_vector shape = {}".format(context_vector.shape))
        
        return context_vector, attention_weights


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units=256):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query shape = [batch_size, seq_length, hidden_num * 2]
        # values shape = [batch_size, seq_length, hidden_num * 2]
        
        # encoder_image_features shape [batch_size, 64, encoder_image_dim]
        # caption shape [batch_size, caption_embedding_dim]

        # query shape = [batch_size, query_emb_dim]
        # values shape = [batch_size, seq_length, hidden_size]
        # hidden_with_time_axis shape == [batch_size, 1, query_emb_dim]
        # broadcast addition along the time axis to calculate the score
        #query_with_time_axis = tf.expand_dims(query, 1)  # [batch_size, 1, query_emb_dim]

        # score shape == (batch_size, seq_length, 1)
        #print("query shape = %s" % str(query.shape))
        #print("values shape = %s" % str(values.shape))
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
        #print("score shape = %s" % str(score.shape))

        # attention_weights shape == [batch_size, seq_length, 1]
        attention_weights = tf.nn.softmax(score, axis=1)
        #print("att_weights shape = %s" % str(attention_weights.shape))
        
        # context_vector shape after sum == [batch_size, hidden size]
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        #print("att_result shape = %s" % str(context_vector.shape))

        return context_vector, attention_weights


class AttentionLayer(tf.keras.Model):
    def __init__(self, units, method='concat'):
        super(AttentionLayer, self).__init__()
        # TODO: Three types of score function
        self.method = method
        self.W_a = tf.keras.layers.Dense(units)
        self.v_a = tf.keras.layers.Dense(1)
    
    def call(self, dec_h_t, enc_h_s):
        """
        Args:
            dec_h_t: current target state (batch_size, 1, units)
            enc_h_s: all source states (batch_size, seq_len, units)

        Returns:
            context_vector: (batch_size, units)
        """
        
        # concat_h = tf.concat([dec_h_t, enc_h_s], axis=1)
        # concat_h = tf.reshape(concat_h, [concat_h.shape[0] * concat_h.shape[1], concat_h.shape[2]])
        # print('concat_h shape:', concat_h.shape)
        
        # score shape == (batch_size, seq_len, 1)
        if self.method == 'concat':
            score = self.v_a(tf.nn.tanh(self.W_a(dec_h_t + enc_h_s)))
        elif self.method == 'general':
            score = tf.matmul(self.W_a(enc_h_s), dec_h_t, transpose_b=True)
        elif self.method == 'dot':
            score = tf.matmul(enc_h_s, dec_h_t, transpose_b=True)
        
        # a_t shape == (batch_size, seq_len, 1)
        a_t = tf.nn.softmax(score, axis=1)
        
        # TODO: replace matmul operator with multiply operator
        # tf.matmul(a_t, enc_h_s, transpose_a=True) -> a_t * enc_h_s
        # result shape after * operation: (batch_size, seq_len, units)
        
        # (batch_size, 1, units)
        # context_vector shape == (batch_size, units)
        context_vector = tf.reduce_sum(a_t * enc_h_s, axis=1)
        
        return context_vector


if __name__ == "__main__":
    att = SelfAttention(10)
    att(tf.ones((32, 128)))
    pass