# -*- encoding:utf-8 -*-
import json
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

os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error



# 加载模型
#from keras.model import load_model
from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras.callbacks import Callback
from on_lstm_keras import ONLSTM
import pyrouge

# 加载模型
# define parameters
max_seq_length = 50  # sentence中最大长度,Tree induction模型中最大是200
max_sent_num = 45    # 一个document中最多sentences的数目，Tree induction模型中最大是100
batch_size = 100     # Tree induction中batch_size是10000
word_size = 128     # embedding的维度，为了后续可能使用glove
num_levels = 16     # 为了计算chunk size
x_in = Input(shape=(None,), dtype='int32') # 句子输入
x = x_in

vocab_path = "./data/spm.cnndm.model"
spm = sentencepiece.SentencePieceProcessor()
spm.Load(vocab_path)
vocab_size = len(spm)

x = Embedding(vocab_size, word_size)(x)
x = Dropout(0.25)(x)
onlstms = []

for i in range(3):
    onlstm = ONLSTM(word_size, num_levels, return_sequences=True, dropconnect=0.25)
    onlstms.append(onlstm)
    x = onlstm(x)

x = Dense(vocab_size, activation='softmax')(x)

lm_model = Model(x_in, x)
lm_model.load_weights("./best_model.weights")
#lm_model.add_loss(loss)
#lm_model.compile(optimizer='adam')

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
def rouge_output():
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


def display_tree(input_path):
    output_path = "./tree_display.json"
    output_f = open(output_path,'a+')
    with open(input_path,'r') as f:
        for line in f.readlines():
            sent_list = nltk.sent_tokenize(line)
            res = parse_sent(sent_list)
            res_diplay = json.dumps(res, indent=4, ensure_ascii=False)
            print(res_diplay)
            output_f.write(res_diplay)


input_path = "./test.txt"
display_tree(input_path)
