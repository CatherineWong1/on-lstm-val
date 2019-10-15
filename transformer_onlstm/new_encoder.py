# -*- encoding:utf-8 -*-
"""
0926
Version1.0 encoder全部为LSTM
0930
Version2.0 encoder可选为LSTM 和Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.autograd import Variable


class Encoder(nn.Module):
    """
    初始化sentence-level的encoder
    即输入一个para，将其里面的

    """
    def __init__(self, parser):
        self.layers = parser.num_layers
        self.num_directions = 2
        self.sent_enc_size = parser.sent_enc_size
        assert parser.sent_enc_size % self.num_directions == 0
        self.hidden_size = self.sent_enc_size // self.num_directions
        input_size = 300

        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_size,self.hidden_size,num_layers=self.layers,
                           dropout=parser.sent_dropout,bidirectional=True)

        if parser.sent_dropout > 0:
            self.dropout = nn.Dropout(parser.sent_dropout)
        else:
            self.dropout = None

    def forward(self, input, hidden=None):
        """
        input: (srcBatch, lengths)
        """
        print("Entering Encoder forward function")
        word_emb = input[0] # LSTM
        print("Word embeding shape is {}".format(word_emb.size()))
        lengths = input[1].view(-1).tolist()
        emb = pack(word_emb,lengths)
        outputs, (hidden_t, cell_t) = self.rnn(emb)
 
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        if self.dropout is not None:
            outputs = self.dropout(outputs)
            hidden_t = self.dropout(hidden_t)

        # hidden_t的shape(layers*2, batch*max_doc_len,hidden_size)
        return hidden_t, outputs


class DocumentEncoder(nn.Module):
    def __init__(self, parser):
        self.layers = parser.num_layers
        self.doc_len = parser.max_doc_len
        self.num_directions = 2
        self.doc_enc_size = parser.doc_enc_size
        assert parser.doc_enc_size % self.num_directions == 0
        self.hidden_size = self.doc_enc_size // self.num_directions
        input_size = 512

        super(DocumentEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                          num_layers=self.layers,
                          dropout=parser.doc_dropout,
                          bidirectional=True)
        if parser.doc_dropout > 0:
            self.dropout = nn.Dropout(parser.doc_dropout)
        else:
            self.dropout = None

    def forward(self, input, hidden=None):
        """
        得到Dcoument-level的embedding
        :param input: (sentence-level tensor, doc_lengths)
        :param hidden:
        :return:
        """
        print("Entering DocumentEncoder forward function")
        sentence_tensors = input[0].view(-1, self.doc_len, input[0].size(1))  # (batch, doc_len, embedding_dim)
        sentence_tensors = sentence_tensors.transpose(0,1).contiguous() # (doc_len,batch,embedding_dim)
        lengths = input[1]
        emb = pack(sentence_tensors, lengths.tolist())
        outputs, (hidden_t,cell_t) = self.rnn(emb)
        
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        if self.dropout is not None:
            outputs = self.dropout(outputs)
            hidden_t = self.dropout(hidden_t)
        print("Finish DocumentEncoder forward function")
        return outputs, hidden_t


class TitleGen(nn.Module):
    def __init__(self, sent_encoder, doc_encoder):
        super(TitleGen, self).__init__()
        self.sent_encoder = sent_encoder
        self.doc_encoder = doc_encoder

    def encode_document(self,src, indices):
        """
        Encode the document
        :param src: (src_batch,lenghts,doc_lengths)
        :param indices: indices：每一个sentence的index
        :return: doc_hidden,doc_context,doc_sent_mask
        """
        print("Entering TitleGen encode_document function")
        enc_hidden, context = self.sent_encoder(src)
        # sentence_tensors的shape变为(batch*max_doc_len,seq_lengths*embedding_dim)
        sentence_tensors = enc_hidden.transpose(0,1).contiguous().view(enc_hidden.size(1),-1)
        # 将indicies按照升序排序后，得到其对应的index，该index就是原始sentence的顺序
        _, restore_index = torch.sort(indices,dim=0)
        sentence_tensors = sentence_tensors.index_select(0,restore_index)
        print(sentence_tensors.size())
        doc_hidden,doc_context = self.doc_encoder((sentence_tensors,src[2]))

        return doc_hidden,doc_context

    def forward(self, input):
        print("TitleGen forward function")
        doc_hidden, doc_ontext = self.encode_document(input[0],input[2])

        return doc_hidden,doc_ontext





