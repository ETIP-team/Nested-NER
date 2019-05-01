# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-03

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from _4_29_bilstm_attention_connect_previous_decoder_info_add_emb.control_config import Config
from _4_29_bilstm_attention_connect_previous_decoder_info_add_emb.attention_neww2vmodel import geniaDataset


class AttentionNestedNERModel(nn.Module):
    def __init__(self, config: Config, word_dic: geniaDataset):
        super(AttentionNestedNERModel, self).__init__()
        self.config = config
        # basic setting
        embedding_dim = config.embedding_dim
        hidden_units = config.hidden_units
        encode_bi_flag = config.encode_bi_flag
        encode_num_layers = config.encode_num_layers
        decode_num_layers = config.decode_num_layers
        self.classes_num = config.classes_num
        self.linear_hidden_units = config.linear_hidden_units

        # model setting:
        self.encode_lstm = nn.LSTM(embedding_dim, hidden_units, encode_num_layers, bidirectional=encode_bi_flag)
        self.embedding = nn.Embedding.from_pretrained(word_dic.weight)
        self.embedding.weight.requires_grad = True

        self.decode_hidden_units = hidden_units * 2 if encode_bi_flag else hidden_units
        # max nested level one hot!
        # * 3 is VNER.

        # rectify one: do not use hot one control nested flag anymore.
        if self.config.add_control_flag:
            if self.config.attention_method != "PLQ" and self.config.attention_method != "concate_before_attention":
                self.decode_lstm = nn.LSTM(self.decode_hidden_units * 4 + self.config.max_nested_level,
                                           hidden_units * 2, decode_num_layers)
            else:
                self.decode_lstm = nn.LSTM(self.decode_hidden_units * 4, hidden_units * 2, decode_num_layers)

        if self.config.attention_method == "concate":
            self.weight_linear = nn.Linear(self.decode_hidden_units + embedding_dim, 1)
        elif self.config.attention_method == "general":  # general
            self.weight_linear = nn.Linear(self.decode_hidden_units, self.decode_hidden_units, bias=False)
        elif self.config.attention_method == "PLQ":  # method 2
            self.P_weight = nn.Parameter(t.randn(self.decode_hidden_units, self.config.max_nested_level))
            self.Q_weight = nn.Parameter(t.randn(self.config.max_nested_level, self.decode_hidden_units))
        elif self.config.attention_method == "concate_before_attention":  # method 3
            self.weight_linear = nn.Linear(self.decode_hidden_units + self.config.max_nested_level,
                                           self.decode_hidden_units + self.config.max_nested_level, bias=False)
        # nested flag weight
        else:  # dot method. should not go here.
            pass

        # self.linear1 = nn.Linear(self.decode_hidden_units, self.linear_hidden_units)
        # 4-12 one linear layer.
        # self.linear2 = nn.Linear(self.linear_hidden_units, self.classes_num)
        self.linear2 = nn.Linear(self.decode_hidden_units, self.classes_num)

        self.optimizer = None
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # rectify two:
    def compute_context_input(self, s_i: Variable, seqs: Variable, previous_s: Variable, h: Variable, time: int,
                              current_nested_level: int) -> Variable:
        """
        To compute context input in time t+1 in different compute type.

        :param s_i: decoder output states in time t. shape: [1, num_batch, hidden_size]
        :param h: encoder shape: [seq_len, num_batch, embedding_dim]
        :param time: time t + 1 index.
        :return:
        """
        # context_input =

        seq_len = h.shape[0]
        num_batch = h.shape[1]

        # one hot
        one_hot_control_nested_np = np.zeros((1, num_batch, self.config.max_nested_level))
        one_hot_control_nested_np[:, :, current_nested_level] = 1
        control_nested_tensor = t.Tensor(
            one_hot_control_nested_np)  # [seq_len = 1, num_batch, one_hot nested_level]
        # control_nested_tensor = t.Tensor(
        #     np.ones((1, num_batch, 1)) * control_nested_level)

        control_nested_tensor = Variable(control_nested_tensor.cuda()) if self.config.cuda else Variable(
            control_nested_tensor)

        if self.config.attention_method == "general":
            weight_input = self.weight_linear(h.permute(1, 0, 2))  # [num_batch, seq_len, hidden_size]
            weight_input = t.bmm(weight_input, s_i.permute(1, 2, 0))  # [num_batch, seq_len, 1]
            norm_weight = F.softmax(weight_input, dim=1)  # normalize in dimension seq_len

        elif self.config.attention_method == "dot":

            weight_input = t.bmm(s_i.permute(1, 0, 2), h.permute(1, 2,
                                                                 0))  # [num_batch, 1, hidden_size] [num_batch, emb_dim,seq_len]  = [num_batch, 1, seq_len]
            weight_input = weight_input.permute(0, 2, 1)  # [num_batch, seq_len, 1]
            norm_weight = F.softmax(weight_input, dim=1)

        elif self.config.attention_method == "concate":
            s_compute = s_i.expand(seq_len, num_batch, -1)  # [seq_len, num_batch, hidden_size]
            weight_input = t.cat([s_compute, h], 2)  # [seq_len, num_batch, hidden_size + embedding_dim]
            norm_weight = F.softmax(self.weight_linear(weight_input), dim=0).permute(1, 0,
                                                                                     2)  # [num_batch, seq_len, 1]

        elif self.config.attention_method == "PLQ":  # method 2  # to test.
            nested_matrix = Variable(t.zeros(self.config.max_nested_level, self.config.max_nested_level))
            nested_matrix = nested_matrix.cuda() if self.config.cuda else nested_matrix
            nested_matrix[current_nested_level, current_nested_level] = 1
            weight_matrix = self.P_weight.mm(nested_matrix)

            # h.permute(1, 0, 2)  # [num_batch, seq_len, hidden_size]
            weight_matrix = weight_matrix.mm(self.Q_weight).unsqueeze(0).expand(num_batch, self.decode_hidden_units,
                                                                                self.decode_hidden_units)  # [num_batch, decode_hidden_units,decode_hidden_units]
            weight_input = t.bmm(h.permute(1, 0, 2), weight_matrix)  # [num_batch, seq_len, decode_hidden_units]
            weight_input = t.bmm(weight_input, s_i.permute(1, 2, 0))  # [num_batch, seq_len, 1]
            norm_weight = F.softmax(weight_input, dim=1)

        elif self.config.attention_method == "concate_before_attention":  # method 3
            s_compute = t.cat([s_i, control_nested_tensor], 2)  # [1, num_batch, decode_hidden_size + max_nested]
            h_compute = t.cat([h, control_nested_tensor.expand(seq_len, num_batch,
                                                               self.config.max_nested_level)],
                              2)  # [seq_len, num_batch, decode_hidden_size + max_nested]

            weight_input = self.weight_linear(h_compute.permute(1, 0, 2))  # [num_batch, seq_len, hidden_size]
            weight_input = t.bmm(weight_input, s_compute.permute(1, 2, 0))  # [num_batch, seq_len, 1]
            norm_weight = F.softmax(weight_input, dim=1)  # normalize in dimension seq_len

        else:
            raise KeyError("attention compute method not right.")

        context_input = t.bmm(h.permute(1, 2, 0),
                              norm_weight)  # [num_batch, embedding_dim, seq_len] [num_batch, seq_len, 1] = [num_batch, embedding_dim, 1]

        h_i = h[time, :, :].unsqueeze(0).permute(1, 2, 0)  # [num_batch, embedding_dim, 1]
        # rectify four
        seqs_i = seqs[time, :, :].unsqueeze(0).permute(1, 2, 0)  # [num_batch, embedding_dim, 1]
        previous_s_i = previous_s[time, :, :].unsqueeze(0).permute(1, 2, 0)  # [num_batch, embedding_dim, 1]
        context_input = t.cat([context_input, h_i, seqs_i, previous_s_i],
                              1)  # [num_batch, embedding_dim * 4, seq_len = 1]

        context_input = context_input.permute(2, 0, 1)  # [seq_len = 1, num_batch, embedding * 2]

        if self.config.add_control_flag:
            if self.config.attention_method != "PLQ" and self.config.attention_method != "concate_before_attention":
                context_input = t.cat([context_input, control_nested_tensor], 2)  # add in third dim.

        return context_input

    def compute_decoder_init_state(self, h_t, s_t, num_batch, decode_num_layers, decode_hidden_size):
        if self.config.encoder_decoder_connection:  # todo test.
            s_i = h_t.view(1, num_batch, -1).expand(decode_num_layers, num_batch, -1)
            cell_state_i = s_t.view(1, num_batch, -1).expand(decode_num_layers, num_batch, -1)
        else:
            s_i = np.zeros((decode_num_layers, num_batch, decode_hidden_size))
            cell_state_i = np.zeros((decode_num_layers, num_batch, decode_hidden_size))
            s_i = Variable(t.Tensor(s_i))
            cell_state_i = Variable(t.Tensor(cell_state_i))
            if self.config.cuda:
                s_i = s_i.cuda()
                cell_state_i = cell_state_i.cuda()
        return s_i, cell_state_i

    def forward(self, seqs, seq_max_nested_level):  # todo modify this forward to
        """seqs: Tensor for word idx."""

        seqs = self.embedding(seqs).permute(1, 0, 2)  # [seq_len, num_batch, embedding_dim]
        seq_len = seqs.shape[0]
        num_batch = seqs.shape[1]
        decode_num_layers = self.decode_lstm.num_layers
        decode_hidden_size = self.decode_lstm.hidden_size

        h, (h_t, s_t) = self.encode_lstm.forward(seqs)  # [seq_len, num_batch, embedding_dim]
        # h = h.permute(1, 0, 2)  # [num_batch, seq_len, embedding_dim]
        # seqs be one cause decode input only one
        # initial i = 0.

        # if self.config.encoder_decoder_connection:  # todo test.
        #     s_i = h_t.view(1, num_batch, -1).expand(decode_num_layers, num_batch, -1)
        #     cell_state_i = s_t.view(1, num_batch, -1).expand(decode_num_layers, num_batch, -1)
        # else:
        #     s_i = np.zeros((decode_num_layers, num_batch, decode_hidden_size))
        #     cell_state_i = np.zeros((decode_num_layers, num_batch, decode_hidden_size))
        #     s_i = Variable(t.Tensor(s_i))
        #     cell_state_i = Variable(t.Tensor(cell_state_i))
        #     if self.config.cuda:
        #         s_i = s_i.cuda()
        #         cell_state_i = cell_state_i.cuda()
        s_i, cell_state_i = self.compute_decoder_init_state(h_t, s_t, num_batch, decode_num_layers, decode_hidden_size)

        output_list = []
        # decode input seq_len must 1
        previous_s_list = []
        for control_nested_level in range(seq_max_nested_level):
            if control_nested_level == 0:  # fisrt layer:
                previous_s = h
            else:
                wait = True

            if not self.config.level_connection:
                s_i, cell_state_i = self.compute_decoder_init_state(h_t, s_t, num_batch, decode_num_layers,
                                                                    decode_hidden_size)
            one_nested_level_output_list = []
            for context_index in range(seq_len):
                s_compute = s_i[-1].unsqueeze(0)  # get last layer [1, num_batch, hidden_size]

                # todo context_input add control info.

                context_input = self.compute_context_input(s_compute, seqs, previous_s, h,
                                                           context_index,
                                                           control_nested_level)  # [seq_len = 1, num_batch, embedding * 3]

                # include h_decode_s_previous.
                # context_input = t.cat([context_input, s_compute, control_nested_tensor],
                #                       2)  # add in third dim. 4-9 VNER

                # save one time output and update the cell state.

                one_time_output, (s_i, cell_state_i) = self.decode_lstm.forward(context_input, (s_i, cell_state_i))
                one_nested_level_output_list.append(one_time_output)
                previous_s_list.append(s_i)  # rectify three
            output_list.append(
                t.cat(one_nested_level_output_list).unsqueeze(0))  # seq_len, batch_num, decode_hidden_size

            previous_s = t.cat(previous_s_list, 0)  # [seq_len, num_batch, embedding_dim]

        output = t.cat(output_list, 0)  # [nested_level, seq_len, batch_num, decode_hidden_size]

        # output = F.relu(self.linear1(output))  # forward 4-12 one linear.
        output = self.linear2(output)  # [seq_len, batch_num, bio_classes_num]
        return output.reshape(-1, self.classes_num)

    def calc_loss(self, predict_result, labels):
        """predict result is FloatTensor
           labels is LongTensor!"""

        loss = self.cross_entropy_loss(predict_result, labels)
        return loss

# if __name__ == '__main__':
#     model = AttentionNestedNERModel(Config())
#     path = "/"
