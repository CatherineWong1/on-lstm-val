# -*- encoding:utf-8 -*-

import nltk
import json
import os
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error 



"""
数据预处理的思路：
1. 首先读取每一行，一行就是一个document
2. 构建单词词典
3. 构建document的张量
"""

# define parameters
max_seq_length = 36  # sentence中最大长度
max_sent_num = 32    # 一个document中最多sentences的数目
batch_size = 32
word_size = 128     # embedding的维度，为了后续可能使用glove
num_levels = 16     # 为了计算chunk size


def text_generator():
    """
    一行就是一个标题下的所有句子的集合。
    :return:
    """
    with open('documents.txt') as f:
        for t in f:
            yield t.strip().decode('utf-8')


# 构建词典
if os.path.exists('onlstm_config.json'):
    words, id2word, word2id = json.load(open('onlstm_config.json'))
    id2word = {int(i):j for i,j in id2word.items()}
else:
    words = {}
    for s in text_generator():
        sent_list = nltk.sent_tokenize(s)
        for sent in sent_list:
            for w in nltk.word_tokenize(sent):
                words[w] = words.get(w, 0) + 1

    # 0: padding, 1: unk
    words = {i: j for i, j in words.items() if j >= 1}
    with open("vocab.txt", 'w') as f:
        for i, j in enumerate(words):
            f.write(j)
            f.write('\n')
    id2word = {i + 2: j for i, j in enumerate(words)}
    word2id = {j: i for i, j in id2word.items()}
    json.dump([words, id2word, word2id], open('onlstm_config.json', 'w'))


# 构建训练集
"""
Input representationd的思路：
最终的Input tensor shape为（batch_size, sentence_num, word_size)
output tensor中可以抽取出来的层级信息结点，每一个结点代表一个sentence
sentence-level是由word-level构成的
一个sentence的tensor shape为（max_seq_length, word_size)
一个document的tensor shape为（sentence_num,max_seq_length, word_size)
多个document的tensor shape为（batch_size,sentence_num, max_seq_length, word_size)
对一个batch的document的 axis=2进行sum 即可得到shape为(batch_size, sentence_num,word_size)
"""
def data_generator():
    batch_doc_set = []
    while True:
        for s in text_generator():
            sent_list = nltk.sent_tokenize(s)
            sent_index_set = []
            for sent in sent_list:
                word_list = nltk.word_tokenize(sent)[:max_seq_length]
                word_index = [word2id.get(w, 1) for w in word_list]
                sent_index_set.append(word_index)

            # 限制sentence的最大数量
            # 限制sentence的最大数量
            if len(sent_index_set) >= max_sent_num:
                sent_index_set = sent_index_set[:max_sent_num]
            else:
                while len(sent_index_set) < max_sent_num:
                    sent_index_set.append([0]*max_seq_length)
            # 对齐所有sentence的长度
            sent_index_set = [x + [0] * (max_seq_length - len(x)) for x in sent_index_set]
            batch_doc_set.append(sent_index_set)
            if len(batch_doc_set) == batch_size:
                temp_tensor = np.array(batch_doc_set)
                batch_tensor = np.sum(temp_tensor,axis=2)
                batch_tensor = batch_tensor/max_sent_num
                yield batch_tensor, None
                batch_doc_set = []

train_data = data_generator()


from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras.callbacks import Callback
from on_lstm_keras import ONLSTM


x_in = Input(shape=(None,), dtype='int32') # 句子输入
x = x_in

x = Embedding(len(words)+2, word_size)(x)
x = Dropout(0.25)(x)
onlstms = []

for i in range(3):
    onlstm = ONLSTM(word_size, num_levels, return_sequences=True, dropconnect=0.25)
    onlstms.append(onlstm)
    x = onlstm(x)

x = Dense(len(words)+2, activation='softmax')(x)

x_mask = K.cast(K.greater(x_in[:, :-1], 0), 'float32')
loss = K.sum(K.sparse_categorical_crossentropy(x_in[:, 1:], x[:, :-1]) * x_mask) / K.sum(x_mask)

lm_model = Model(x_in, x)
lm_model.add_loss(loss)
lm_model.compile(optimizer='adam')


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            lm_model.save_weights('./best_model.weights')


evaluator = Evaluate()

lm_model.fit_generator(train_data,
                       steps_per_epoch=100,
                       epochs=100,
                       callbacks=[evaluator])


lm_f = K.function([x_in], [onlstms[0].distance])


def build_tree(depth, sen):
    """该函数直接复制自原作者代码
    """
    assert len(depth) == len(sen)
    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = np.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def parse_sent(s):
    eval_index = []
    sent_list = nltk.sent_tokenize(s)
    sent_list = sent_list[:max_sent_num]
    for i in range(len(sent_list)):
        eval_index.append(i)
    eval_index = np.array(eval_index)
    sl = lm_f([eval_index])[0][0][1:]
    # 用json.dumps的indent功能，最简单地可视化效果
    return json.dumps(build_tree(sl, eval_index), indent=4, ensure_ascii=False)


# 读入测试文件
with open("test.txt") as f:
    for item in f:
        s = item.strip()
        res = parse_sent(s)
        print(res)
