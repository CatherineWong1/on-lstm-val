# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import bert.modeling as modeling
import bert.optimization as optimization
import six
import tensorflow as tf

import nltk
import pickle
import os
import numpy as np
import glob
import sentencepiece
import shutil
import random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error


from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras.callbacks import Callback
from on_lstm_keras import ONLSTM

import pyrouge



# 导入数据集
def load_dataset():
    """
    加载CNN/DM的数据集
    :return:
    """

    # assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pickle_file, corpus_type):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
            # print('Loading %s dataset from %s, number of examples: %d' %
            # (corpus_type, pickle_file, len(dataset)))
            return dataset

    # Sort the glob output by file name (by increasing indexes).
    dataset_path = "./data/"
    corpus_type = "train_sum"
    pts = sorted(glob.glob(dataset_path + corpus_type + '.[0-9]*.pk'))

    shuffle = False
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            print(pt)
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = dataset_path + corpus_type + '.pk'
        yield _lazy_dataset_loader(pt, corpus_type)


vocab_size = 28996


# 使用bert构建Document-level的向量
def bert_embedding(mode,input_ids, input_mask, segment_ids):
    # 1. 导入bert pretrained model
    bert_config_file = "./bert/bert-cased/bert_config.json"
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    # 2. 建立输出目录，不一定需要，待定
    output_dir = "./output"
    tf.gfile.MakeDirs(output_dir)
    is_training = mode

    # input_ids的shape: batch_size ,seq_length
    # input_mask,segment_ids的shape和上面一样，均为batch_size, seq_length
    # input_ids: 对应数据集中的src
    # input_mask：The mask has 1 for real tokens and 0 for padding tokens. Only real
    # segement_ids: 对应数据集中的segs
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids)
    # final hidden的shape(batch_size, seq_length, hidden_size)
    final_hidden = model.get_sequence_output()
    return final_hidden


# define parameters
max_seq_length = 512
batch_size = 30


def create_bert_inputs():
    for data_list in load_dataset():
        for s in data_list:
            input_ids = s['src']
            input_mask = [1] * len(input_ids)
            segment_ids = s['segs']
            # 对齐所有sentence的长度
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            yield input_ids, input_mask,segment_ids


def data_generator():
    """
    思路：完全可以参考BertSum
    用了BertSum中已经处理好的数据集，一条数据即是一篇document，其最大长度为512
    因此只需要先Load进来batch_size大小的数据集
    分别构成：input_ids,input_maks,segments_id
    max_seq_length = 512，如果不到全部补0
    :return:
    """
    batch_input_ids = []
    batch_input_mask = []
    batch_segment_ids = []
    while True:
        for bert_input in create_bert_inputs():
            tmp_input_ids = bert_input[0]
            tmp_input_mask = bert_input[1]
            tmp_segment_ids = bert_input[2]

            batch_input_ids.append(tmp_input_ids)
            batch_input_mask.append(tmp_input_mask)
            batch_segment_ids.append(tmp_segment_ids)

            if (len(batch_input_ids) == batch_size) and (len(batch_input_mask) == batch_size) and (len(batch_segment_ids) == batch_size):
                mode =False
                input_ids = tf.constant(batch_input_ids)
                input_mask = tf.constant(batch_input_mask)
                segment_ids = tf.constant(batch_segment_ids)
                document_embeddings = bert_embedding(mode, input_ids, input_mask, segment_ids)
                document_embeddings = K.get_value(document_embeddings)
                yield document_embeddings, None
                batch_input_ids = []
                batch_input_mask = []
                batch_segment_ids = []



train_data = data_generator()

# define training parameters
word_size = 768 # equal with max_seq_length
num_levels = 32
# It's not fixed, because we randomly choose one dataset and calculate the maximum length of src txt，
# we found the number is 131, meanwhile the minimum length is 5
max_sent_num = 200


x_in = Input(shape=(max_seq_length,word_size), dtype='float32')  # 句子输入
x = x_in

x = Dropout(0.25)(x)
onlstms = []

for i in range(3):
    onlstm = ONLSTM(word_size, num_levels, return_sequences=True, dropconnect=0.25)
    onlstms.append(onlstm)
    x = onlstm(x)

x = Dense(word_size, activation='softmax')(x)

x_mask = K.cast(K.greater(x_in, 0), 'float32')
loss = K.sum(K.sparse_categorical_crossentropy(x_in, x)) / K.sum(x_mask)
print("Finish calculating loss")


lm_model = Model(x_in, x)
lm_model.add_loss(loss)
lm_model.compile(optimizer='adam')


print("finish compile")

class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            lm_model.save_weights('./best_model.weights')
        # 一个epoch结束，保存结果
        model_path = "./model/{}_epoch.model".format(epoch)
        lm_model.save(model_path)


evaluator = Evaluate()


lm_model.fit_generator(train_data,
                       steps_per_epoch=10000,
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


def parse_sent(sent_list):
    sent_index = []
    # sent_list = nltk.sent_tokenize(s)
    sent_list = sent_list[:max_sent_num]
    for i in range(len(sent_list)):
        sent_index.append(i)
    eval_index = np.array([sent_index])
    sl = lm_f([eval_index])[0][0]
    # 用json.dumps的indent功能，最简单地可视化效果
    # return json.dumps(build_tree(sl, sent_index), indent=4, ensure_ascii=False)
    res = build_tree(sl, sent_index)
    return res


# 预测层级结构并计算rouge分数
def test():
    test_file = "./data/test_sum.0.pk"
    f = open(test_file, 'rb')
    test_data = pickle.load(f)
    tmp_dir = "./rouge"
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")

    for i in range(len(test_data)):
        test_dict = test_data[i]
        sent_list = test_dict['src_txt']
        gold_sum = str(test_dict['tgt_txt'])
        tree_res = parse_sent(sent_list)
        # select candidates
        pred_index = select_candidate(tree_res)
        _pred = []
        for ind in pred_index:
            _pred.append(sent_list[ind])
        pred_sum = '<q>'.join(_pred)
        with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w") as f:
            f.write(pred_sum)
        with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w") as f:
            f.write(gold_sum)

    rouges = cal_rouge(tmp_dir)
    print(rouge_results_to_str(rouges))


def select_candidate(tree_res):
    """
    分析层级结构，并选出前3句话作为摘要
    :param tree_res:
    :return: candidates index list
    """
    candidates_list = []
    for item in tree_res:

        def choose_cand(item, cand_list):
            if len(cand_list) < 3:
                if type(item) == int:
                    cand_list.append(item)
                elif type(item) == list:
                    for sub_item in item:
                        cand_list = choose_cand(sub_item, cand_list)

            return cand_list

        cand_len = len(candidates_list)
        if cand_len < 3:
            candidates_list = choose_cand(item, candidates_list)
        else:
            break
    return candidates_list


def cal_rouge(tmp_dir):
    """
    计算rouge 分数
    :return:
    """
    try:
        temp_dir = "/home/huqian/anaconda2/envs/on_lstm_env/lib/python2.7/site-packages/pyrouge"
        r = pyrouge.Rouge155(rouge_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        # print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)

    finally:
        pass
        # if os.path.isdir(tmp_dir):
        #    shutil.rmtree(tmp_dir)

    return results_dict


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100

        # ,results_dict["rouge_su*_f_score"] * 100
    )


test()


