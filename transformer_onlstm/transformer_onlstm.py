# -*- encoding:utf-8 -*-
import tensorflow as tf
import opennmt
import pickle
import glob
import random
import nltk


# define parameters
max_seq_length = 100
max_sents_num = 10

import transformer_demo
train_data = transformer_demo.get_trans_doc()

from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras.callbacks import Callback
from on_lstm_keras import ONLSTM
import keras


sent_size = 128
num_levels = 16

class TranON(tf.keras.Model):
    def __init__(self):
        super(TranON, self).__init__()
        self.layer1 = ONLSTM(sent_size, num_levels, return_sequences=True, dropconnect=0.25)
        self.layer2 = ONLSTM(sent_size, num_levels, return_sequences=True, dropconnect=0.25)
        self.layer3 = ONLSTM(sent_size, num_levels, return_sequences=True, dropconnect=0.25)
        self.dense = Dense(sent_size, activation='softmax')

    def call(self, inputs):
        print("Entering Call function")
        self.layer1_output = self.layer1(inputs)
        self.layer2_output = self.layer2(self.layer1_output)
        self.layer3_output = self.layer3(self.layer2_output)
        self.final_output = self.dense(self.layer3_output)
        return self.final_output

    def add_loss(self,x,logits):
        x_mask = K.cast(K.greater(x, 0), 'float32')
        loss_value = K.sum(K.categorical_crossentropy(x, logits)) / K.sum(x_mask)
        return loss_value



trans_on = TranON()

"""
class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            trans_on.save_weights('./best_model.weights')
        # 一个epoch结束，保存结果
        model_path = "./model/{}_epoch.model".format(epoch)
        trans_on.save(model_path)


evaluator = Evaluate()

optimizer = tf.train.AdamOptimizer()
loss_history = []
for x,y in train_data:
    logits = trans_on(x)
    #def loss_func():
    #    with tf.GradientTape() as tape:
    #        x_mask = K.cast(K.greater(x, 0), 'float32')
    #        return K.sum(K.categorical_crossentropy(x, logits)) / K.sum(x_mask)

    trans_on.add_loss(x,logits)
    trans_on.compile(optimizer=optimizer,loss=loss_func)

    trans_on.fit(x,y,steps_per_epoch=10000, epochs=100,callbacks=[evaluator],batch_size = 1)


"""


optimizer = tf.train.AdamOptimizer()
loss_history = []
# Training loop
for i in range(100):
    step = 1
    for x, y in train_data:
        outputs = trans_on(x)
        x_mask = K.cast(K.greater(x, 0), 'float32')
        with tf.GradientTape() as tape:
            outputs = trans_on(x)
            loss_value = K.sum(K.categorical_crossentropy(x[:,1:], outputs[:,:-1])) / K.sum(x_mask)
        loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, outputs)
        optimizer.apply_gradients([(grads,outputs)])
        step += 1
        print("Loss at {} Epoch {} step: {}".format(i+1,step, loss_value))


