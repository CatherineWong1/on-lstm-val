# -*- encoding:utf-8 -*-
import json
import os
import random
import re
import pickle
import argparse
import sentencepiece


def get_token_id(spm, sent_sets):
    src = []
    for i in range(len(sent_sets)):
        sent = sent_sets[i]
        sent_id = spm.encode_as_ids(sent)
        src.append(sent_id)

    return src


def preprocess(args):
    """
    split dataset for train dataset and test dataset
    :param args: params of all
    :return:
    """
    file_path = args.file_path
    test_num = args.test_num
    data_path = args.data_path
    vocab_path = args.vocab_path

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(vocab_path)

    file_list = os.listdir(file_path)
    random.shuffle(file_list)
    test_list = file_list[:test_num]
    train_list = file_list[test_num:]
    # train dataset
    create_data(train_list, 'train', data_path, spm, file_path)

    # test dataset
    create_data(test_list, 'test', data_path, spm, file_path)


def create_data(data_list, corpus_type, data_path, spm,file_path):
    
    # create  dataset
    for i in range(len(data_list)):
        print(data_list)
        data_file = file_path + "/" + data_list[i]
        doc_sets = []
        with open(data_file, 'r') as f:
            data = json.load(f)
            for item in data['procedureArray']:
                if item['process'] == 'true':
                    sample_set = []
                    sample_dict = dict()
                    for single in item['stepList']:
                        step = single['step']
                        step = re.sub(r'[^\x00 -\x7F]', ' ', step)
                        step = step.split('\n')
                       
                        sample_set += [temp for temp in step]

                    # 对一个process进行处理
                    src = get_token_id(spm, sample_set)
                    sample_dict['src_txt'] = sample_set
                    sample_dict['src'] = src

                    doc_sets.append(sample_dict)

        new_data = "./{}/{}.{}.pk".format(data_path, corpus_type, i)
        with open(new_data, 'w') as f:
            pickle.dump(doc_sets, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_path", default='./json_data/')
    parser.add_argument("-data_path", default='./train_data/')
    parser.add_argument("-vocab_path", default="./spm.cnndm.model")
    parser.add_argument("-test_num", default=1)

    args = parser.parse_args()
    preprocess(args)
