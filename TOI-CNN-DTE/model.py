import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Transformer import Transformer


class RegionRepresentation(nn.Module):
    def __init__(self, input_size, if_gpu):
        super(RegionRepresentation, self).__init__()
        self.input_size = input_size
        padding = torch.zeros((input_size, 1))
        self.device = torch.device("cuda:0" if if_gpu else "cpu")
        self.register_buffer('padding', padding)

    def forward(self, features, tois):
        # result1 = []
        result = []
        for i in range(len(tois)):
            result.append(self.representation(features[i], tois[i]))

        if len(result) == 0:
            wait = True
        result = torch.cat(result, 0) if len(result) > 0 else Variable(torch.Tensor())
        return result, np.cumsum([len(s) for s in tois]).astype(np.int32)

    def representation(self, feature, tois):
        start = tois[:, 0]
        end = tois[:, 1]
        half_len = (end - start + 1) // 2
        # padding = torch.zeros((self.input_size, 1)).to(self.device)
        cumsum = torch.cat([Variable(self.padding, requires_grad=False), torch.cumsum(feature, 1)], dim=1)
        # cumsum = torch.cat([Variable(padding, requires_grad=False), torch.cumsum(feature, 1)], dim=1)
        # return torch.cat([self.WeightedAvg(cumsum, start, start + half_len), self.WeightedAvg(cumsum, end - half_len, end)]).t()
        return torch.cat([feature[:, start], self.WeightedAvg(cumsum, start, end), feature[:, end - 1]]).t()
        # return torch.cat([feature[:, start], self.WeightedAvg(cumsum, start, start + half_len),
        #                   self.WeightedAvg(cumsum, end - half_len, end), feature[:, end - 1]]).t()

    def WeightedAvg(self, cumsum, start, end):
        boundary_len = Variable(torch.FloatTensor(end - start), requires_grad=False).to(self.device)

        return (cumsum[:, end] - cumsum[:, start]) / boundary_len



class ROIPooling(nn.Module):
    def __init__(self, config):
        super(ROIPooling, self).__init__()
        self.pooling = nn.AdaptiveMaxPool2d((config.pooling_out_size, 1))
        self.config = config

    def forward(self, features, tois):
        result = []
        for i in range(len(tois)):
            for j in range(tois[i].shape[0]):
                result.append(self.pooling(features[i].unsqueeze(0)[:, :, tois[i][j][0]: tois[i][j][1]]).view(1, -1))

        if len(result) == 0:
            wait = True
        result = torch.cat(result, 0) if len(result) > 0 else torch.Tensor()
        return result, np.cumsum([len(s) for s in tois]).astype(np.int32)


class ChannelAttention(nn.Module):
    def __init__(self, channel, droupout=0.5, reduction=4, multiply=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.Dropout(droupout), nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                )
        self.multiply = multiply
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.multiply == True:
            return x * y
        else:
            return y


class SpatialAttention(nn.Module):
    def __init__(self, h, droupot=0.5, reduction=12, multiply=True):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(h, h // reduction),
                nn.Dropout(droupot),
                nn.ReLU(inplace=True),
                nn.Linear(h // reduction, h),
                nn.Softmax()
                )
        self.multiply = multiply

    def forward(self, x):
        b, c, h, _ = x.size()
        y = self.avg_pool(torch.transpose(x, 1, 2)).view(b, h)
        y = self.fc(y).view(b, 1, h, 1)
        if self.multiply == True:
            return x * y
        else:
            return y


class TOICNN(nn.Module):
    def __init__(self, config):
        super(TOICNN, self).__init__()
        self.config = config
        self.outfile = None

        self.input_size = config.word_embedding_size
        if self.config.if_pos:
            self.pos_embed = nn.Embedding(self.config.pos_tag_kinds, self.config.pos_embedding_size)
            self.input_size += self.config.pos_embedding_size
        if self.config.if_char:
            self.char_embed = nn.Embedding(self.config.char_kinds, self.config.char_embedding_size)
            self.char_bilstm = nn.LSTM(self.config.char_embedding_size, self.config.char_embedding_size,
                                       num_layers=1, batch_first=True, bidirectional=True)
            self.input_size += (2 * self.config.char_embedding_size)
        self.word_embed = nn.Embedding(self.config.word_kinds, self.config.word_embedding_size)

        if self.config.if_transformer is not None and self.config.if_transformer:
            # self.transformer = Transformer(d_model=self.input_size, N=config.in_channels - 1,
            #                                bidirectional=self.config.if_bidirectional)
            self.transformer = Transformer(d_model=self.input_size, N=self.config.N, h=self.config.h,
                                           bidirectional=self.config.if_bidirectional)

        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=config.in_channels, out_channels=self.config.feature_maps_size,
            nn.Conv2d(in_channels=1, out_channels=self.config.feature_maps_size,
                      kernel_size=(self.config.kernel_size, self.input_size),
                      # kernel_size=(self.config.kernel_size, self.input_size * 2),
                      padding=(int(self.config.kernel_size / 2), 0)),
            nn.ReLU())
        if self.config.if_c_attention:
            self.channelAttention = ChannelAttention(self.config.feature_maps_size)
        if self.config.if_s_attention and self.config.if_sent_padding:
            self.spatialAttention = SpatialAttention(186)
        self.region_representation = RegionRepresentation(self.config.feature_maps_size, self.config.if_gpu)
        # self.region_representation = RegionRepresentation(self.input_size, self.config.if_gpu)
        # self.toi_pool = ROIPooling(self.config)
        # self.flatten_feature = self.config.feature_maps_size * self.config.pooling_out_size
        # self.flatten_feature = self.input_size * 3
        self.flatten_feature = self.config.feature_maps_size * 3
        self.flatten_feature2 = self.config.feature_maps_size * 3

        self.cls_fc1 = nn.Sequential(
            # nn.Linear(self.flatten_feature, self.flatten_feature),
            nn.Linear(self.flatten_feature, self.flatten_feature2),
            nn.Dropout(self.config.dropout), nn.ReLU(),
            # nn.Linear(self.flatten_feature, self.config.label_kinds))
            nn.Linear(self.flatten_feature2, self.config.label_kinds))
        self.nested_fc1 = nn.Sequential(
            # nn.Linear(self.flatten_feature, self.flatten_feature),
            nn.Linear(self.flatten_feature, self.flatten_feature2),
            nn.Dropout(self.config.dropout), nn.ReLU(),
            # nn.Linear(self.flatten_feature, len(self.config.nested_flag)))
            nn.Linear(self.flatten_feature2, len(self.config.nested_flag)))

        self.input_dropout = nn.Dropout(self.config.dropout)
        self.cls_ce_loss = nn.CrossEntropyLoss()
        self.nested_weight = Variable(torch.Tensor(np.array(self.config.nested_weight)))
        self.nested_ce_loss = nn.CrossEntropyLoss(weight=self.nested_weight)

    def forward(self, mask_batch, word_batch, char_batch, char_len_batch, pos_batch, toi_batch):
        word_vec = self.word_embed(word_batch)
        if self.config.if_char:
            char_vec = self.char_embed(char_batch)
            char_vec = torch.cat(tuple(char_vec))
            chars_len = np.concatenate(char_len_batch)
            perm_idx = chars_len.argsort(0)[::-1].astype(np.int32)
            back_idx = perm_idx.argsort().astype(np.int32)
            pack = pack_padded_sequence(char_vec[perm_idx], chars_len[perm_idx], batch_first=True)
            lstm_out, (hid_states, cell_states) = self.char_bilstm(pack)
            char_vec = torch.cat(tuple(hid_states), 1)[back_idx]
            word_vec = torch.cat([word_vec, char_vec.view(word_vec.shape[0], word_vec.shape[1], -1)], 2)
        if self.config.if_pos:
            pos_vec = self.pos_embed(pos_batch)
            word_vec = torch.cat([word_vec, pos_vec], 2)
        word_vec = self.input_dropout(word_vec)

        t_num = mask_batch.shape[1]
        if self.config.if_transformer:
            word_vec = self.transformer(word_vec, mask_batch.repeat(1, t_num).unfold(1, t_num, t_num))
        if self.config.if_sent_padding:
            word_vec = word_vec.masked_fill(mask_batch.unsqueeze(-1) == 0, torch.tensor(0))

        features = self.conv1(word_vec.unsqueeze(1))
        # features = self.conv1(word_vec)
        # features = word_vec.unsqueeze(1).permute((0, 3, 2, 1))
        if self.config.if_c_attention:
            features = self.channelAttention(features)
        if self.config.if_s_attention and self.config.if_sent_padding:
            features = self.spatialAttention(features)
        # features, toi_section = self.toi_pool(features, toi_batch)
        features, toi_section = self.region_representation(features.squeeze(-1), toi_batch)
        # if features.shape[0] == 0:
        #     s = self.batch_split(features, toi_section)
        #     return s, s
        # features = self.fcs(features)

        # features = self.input_dropout(features)
        cls_s = self.cls_fc1(features)
        nested_s = self.nested_fc1(features)

        # return self.batch_split(cls_s, toi_section), self.batch_split(nested_s, toi_section)
        return cls_s, nested_s, toi_section

    def load_vector(self):
        with open(self.config.get_pkl_path("word2vec"), "rb") as f:
            vectors = pickle.load(f)
            w_v = torch.Tensor(vectors)
            print(f"Loading from {self.config.get_pkl_path('word2vec')}")
            self.word_embed.weight = nn.Parameter(w_v)
            # self.word_embed.weight.requires_grad = False

    def calc_loss(self, cls_s, nested_s, gold_label, gold_nested):
        cls_loss = self.cls_ce_loss(cls_s, gold_label)

        cls = torch.argmax(cls_s, 1)
        mask = (cls != 0)
        # mask = (gold_label != 0)
        # _nested_s = torch.cat(nested_s, 0)[mask]
        _nested_s = nested_s[mask]
        nested_loss = self.nested_ce_loss(_nested_s, gold_nested[mask]) if _nested_s.size(0) != 0 else torch.tensor(0)
        loss = cls_loss + self.config.lambd * nested_loss

        return loss, cls_loss, nested_loss
        # return cls_loss, cls_loss, nested_loss

