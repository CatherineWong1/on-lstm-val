# -*- encoding:utf-8 -*-
import tensorflow as tf
import opennmt
import pickle
import glob
import random
import nltk

tf.enable_eager_execution()


# define parameters
max_seq_length = 100
max_sents_num = 10


# 加入glove
vocab_vectors = {}
glove_path = "./glove_data/glove.840B.300d.txt"
with open(glove_path) as f:
    for i,line in enumerate(f):
        if i % 100000 == 0:
            print("          process line %d" %i)
        s = line.strip()
        word = s[:s.find(' ')]
        vector = s[s.find(' ')+1:]
        vocab_vectors[word] = vector


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
    dataset_path = "./data_demo/"
    corpus_type = "train"
    pts = sorted(glob.glob(dataset_path + corpus_type + '.[0-9]*.pk'))
    shuffle = False
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = dataset_path + corpus_type + '.pk'
        yield _lazy_dataset_loader(pt, corpus_type)


def text_generator():
    """
    返回单个文档数据
    :return:
    """
    for data_list in load_dataset():
        for s in data_list:
            sent_index_set = s['src_txt']
            sent_list = []
            # 按照max_seq_length对齐每一个sentence
            for sent in sent_index_set:
                sent_embed = []
                word_list = nltk.word_tokenize(sent)[:max_seq_length]
                for word in word_list:
                    if word in vocab_vectors:
                        word_vector = map(float, vocab_vectors[word].split())
                    else:
                        word_vector = [0] * 300
                    
                    sent_embed.append(word_vector)

                if len(sent_embed) < max_seq_length:
                    while len(sent_embed) < max_seq_length:
                        sent_embed.append([0] * 300)
                sent_list.append(sent_embed)
            sent_list = pad_single_doc(sent_list)
            
            yield sent_list


def pad_single_doc(sent_list):
    # 按照max_sent_num对齐
    sent_list = sent_list[:max_sents_num]
    while len(sent_list) < max_sents_num:
        pad_sentence = [[0] * 300] * max_seq_length
        sent_list.append(pad_sentence)

    return sent_list


def get_trans_sent():
    # 按照max_sent_num对齐一个document中的sentence数目
    doc_list = text_generator()
    for sent_list in doc_list:
        sents_vector = tf.constant(sent_list)
        doc_length = tf.constant([max_sents_num])
        encoder = opennmt.encoders.SelfAttentionEncoder(num_layers=6)
        # outputs的shape为(max_sents_num,max_sequence_length,num_units),这里max_sents_num即batch_size
        outputs, _, _ = encoder.encode(inputs=sents_vector, sequence_length=doc_length, mode=tf.estimator.ModeKeys.TRAIN)
        yield outputs


def get_trans_doc():
    batch_doc = []
    batch_size = 5
    while True:
        for demo_output in get_trans_sent():
            batch_doc.append(demo_output.numpy())
            print(len(batch_doc))
            if len(batch_doc) == 5:
                docs_tensor = tf.constant(batch_doc)
                docs_shape = docs_tensor.get_shape().as_list()
                """
                docs_tensord的原始shape为（batch_size, max_sents_num, max_seq_length,num_units）
                需要对docs_tensor进行转换，转换过程：
                1. 用PCA的方法，先求出tensor的SVD，左奇异矩阵是我们需要的，对num_units进行压缩
                其shape变为(batch_size，max_sents_num,max_seq_length,max_seq_length)
                2. 将压缩行后的doc_tensor进行reshape
                3. 继续PCA压缩
                """
                # doc_word_vector = tf.reshape(docs_tensor, [batch_size,max_sents_num, max_seq_length * docs_shape[3]])
                s, u, v = tf.linalg.svd(docs_tensor)
                pca = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v[:, :, :100, :], adjoint_b=True))
                new_pca = tf.reshape(pca, [batch_size, max_sents_num, max_seq_length * max_seq_length])
                # 继续降维，指定sentence的sentence-level embedding dim == 128
                new_pca = tf.transpose(new_pca,perm=[0,2,1])
                ss, su, sv = tf.linalg.svd(new_pca)
                new_doc_tensor = tf.matmul(su[:,:128,:], tf.matmul(tf.linalg.diag(ss), sv, adjoint_b=True))
                new_doc_tensor = tf.transpose(new_doc_tensor,perm=[0,2,1])
                encoder = opennmt.encoders.SelfAttentionEncoder(num_layers=6,num_units=128)
                # outputs的shape为(max_sents_num,max_sequence_length,num_units),这里max_sents_num即batch_size
                outputs, _, _ = encoder.encode(inputs=new_doc_tensor, sequence_length=tf.constant([5]), mode=tf.estimator.ModeKeys.TRAIN)
                yield outputs ,None
                batch_doc = []


