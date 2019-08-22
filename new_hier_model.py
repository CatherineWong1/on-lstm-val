# -*- encoding:utf-8 -*-

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


"""
数据预处理的思路：
1. 首先读取每一行，一行就是一个document
2. 构建单词词典
3. 构建document的张量
"""

# define parameters
max_seq_length = 50  # sentence中最大长度,Tree induction模型中最大是200
max_sent_num = 45    # 一个document中最多sentences的数目，Tree induction模型中最大是100
batch_size = 100     # Tree induction中batch_size是10000
word_size = 128     # embedding的维度，为了后续可能使用glove
num_levels = 16     # 为了计算chunk size


def load_dataset():
    """
    加载CNN/DM的数据集
    :return:
    """
    #assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pickle_file, corpus_type):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
            #print('Loading %s dataset from %s, number of examples: %d' %
                    #(corpus_type, pickle_file, len(dataset)))
            return dataset

    # Sort the glob output by file name (by increasing indexes).
    dataset_path = "./data/"
    corpus_type = "train"
    pts = sorted(glob.glob(dataset_path + corpus_type + '.[0-9]*.pk'))
    shuffle=True
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = dataset_path + corpus_type + '.pk'
        yield _lazy_dataset_loader(pt, corpus_type)


def text_generator(data_list):
    """
    返回单个文档数据
    :return:
    """
    #with open('documents.txt') as f:
    #    for t in f:
    #        yield t.strip().decode('utf-8')
    for data_dict in data_list:
        yield data_dict


# 获得词典大小
vocab_path = "./data/spm.cnndm.model"
spm = sentencepiece.SentencePieceProcessor()
spm.Load(vocab_path)
vocab_size = len(spm)


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
        for data_list in load_dataset():
            for s in data_list:
                sent_list = s['src_txt']
                sent_index_set = s['src']
                # 限制sentence的最大数量
                if len(sent_index_set) >= max_sent_num:
                    sent_index_set = sent_index_set[:max_sent_num]
                else:
                    while len(sent_index_set) < max_sent_num:
                        sent_index_set.append([0]*max_seq_length)
                # 对齐所有sentence的长度
                sent_index_set = [x[:max_seq_length] for x in sent_index_set] 
                sent_index_set = [x + [0] * (max_seq_length - len(x)) for x in sent_index_set]
                sent_index_set = add_position(sent_index_set)
                batch_doc_set.append(sent_index_set)
                if len(batch_doc_set) == batch_size:
                    temp_tensor = np.array(batch_doc_set)
                    batch_tensor = np.sum(temp_tensor,axis=2)
                    batch_tensor = batch_tensor/max_seq_length
                    yield batch_tensor, None
                    batch_doc_set = []

def add_position(doc):
    """
    为每一个document增加<s>和</s>
    :param doc: 长度等于max_sent_num的sent_list
    :return: doc:加了start和end的sent_list
    """
    start_tensor = [[1] * max_seq_length]
    end_tensor = [[2] * max_seq_length]
    new_doc = start_tensor + doc + end_tensor
    return new_doc

train_data = data_generator()


from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras.callbacks import Callback
from on_lstm_keras import ONLSTM

import pyrouge

x_in = Input(shape=(None,), dtype='int32') # 句子输入
x = x_in

x = Embedding(vocab_size, word_size)(x)
x = Dropout(0.25)(x)
onlstms = []

for i in range(3):
    onlstm = ONLSTM(word_size, num_levels, return_sequences=True, dropconnect=0.25)
    onlstms.append(onlstm)
    x = onlstm(x)

x = Dense(vocab_size, activation='softmax')(x)

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
    #sent_list = nltk.sent_tokenize(s)
    sent_list = sent_list[:max_sent_num]
    for i in range(len(sent_list)):
        sent_index.append(i)
    eval_index = np.array([sent_index])
    sl = lm_f([eval_index])[0][0]
    # 用json.dumps的indent功能，最简单地可视化效果
    #return json.dumps(build_tree(sl, sent_index), indent=4, ensure_ascii=False)
    res = build_tree(sl, sent_index)
    return res


# 预测层级结构并计算rouge分数
def test():
    test_file = "./data/test_sum.0.pk"
    f = open(test_file,'rb')
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
                    for sub_item in item :
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
        #print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)

    finally:
        pass
        #if os.path.isdir(tmp_dir):
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
