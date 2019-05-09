#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: Transformer.py 
@time: 2019/04/13

@note:
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math
from torch.autograd import Variable


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    Return the position each tgt word is allowed to look at.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def sequent_mask(mask, reverse=False):
    return torch.cat([_.triu().unsqueeze(0) for _ in mask]) if reverse else torch.cat([_.tril().unsqueeze(0) for _ in mask])


def attention(query, key, value, mask=None, dropout=None):
    """
    An attention function can be described as mapping a query and a set of key-value pairs
    to an output. The output is computed as a weighted sum of the values, where the weight
    assigned to each value is computed by a compatibility function of the query with the
    corresponding key.
    Note for the dimension of the query and key vectors are equal.
    The two most commonly used attention functions(are similar in theoretical complexity):
        additive attention : using a feed-forward network with a single hidden layer
        dot-product(multiplicative) attention: much faster and more space-efficient in practice
                                               since it can be implemented using highly
                                               optimized matrix multiplication code
    Return Attention(Q, K, V)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Transformer(nn.Module):
    def __init__(self, N=2, d_model=200, h=4, dropout=0.1, max_len=400, bidirectional=False):
        super(Transformer, self).__init__()
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, dropout=dropout)
        self.bidirectional = bidirectional
        self.model = Encoder(EncoderLayer(d_model, attn, ff, dropout), N)
        if self.bidirectional:
            self.model = clones(self.model, 2)
        # self.position = PositionalEncoding(d_model, dropout, max_len)

    def forward(self, word_embed, mask=None):
        if self.bidirectional:
            # return self.model[0](self.position(word_embed), sequent_mask(mask)) + \
            #        self.model[1](self.position(word_embed, True), sequent_mask(mask, True))
            return self.model[0](word_embed, sequent_mask(mask)) + \
                   self.model[1](word_embed, sequent_mask(mask, True))
            # return torch.cat([self.model[0](word_embed, sequent_mask(mask)),
            #                   self.model[1](word_embed, sequent_mask(mask, True))], -1)
        else:
            # return self.model(self.position(word_embed), mask)
            return self.model(word_embed, mask)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "linear + softmax "

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.fc = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=-1)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # result = [self.norm(x).unsqueeze(1)]
        # result = [x.unsqueeze(1)]
        # result = []
        for layer in self.layers:
            x = layer(x, mask)
            # result.append(self.norm(x).unsqueeze(1))
            # result.append(x.unsqueeze(1))
        return self.norm(x)
        # return torch.cat(result, 1)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        :return LayerNorm（x + Sublayer(x))

        """
        return x + self.dropout(sublayer(self.norm(x)))
        # return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """
    Each layer has two sub-layers:
        multi-head self-attention mechanism : all of the keys, values, queries come from
                                              the output of the previous layer in the encoder
        feed-forward network (position-wise fully connection)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Each layer has three sub-layers:
        masked multi-head self-attention mechanism: all of the keys, values, queries come from
                                                    the output of the previous layer in the encoder
        multi-head src-attention mechanism: the queries come from the previous decoder layer, and the
                                            memory keys and values come from the output of the encoder
        feed-forward network (position-wise fully connection)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param tgt_mask: combined with fact that output embeddings are offset by position, ensures
                         that predictions for position i can depend only on the known outputs at
                         positions less than i
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, head_2, ... head_n) * Wo
        where head_i = Attention(Q * Wq_i, K * Wk_i, V * Wv_i)
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: num of heads(parallel attention layers)
        :param d_model: dimension of input(embedding)

        :parameter linears:
                        Wq_i [d_model, d_k] * h
                        Wk_i [d_model, d_k] * h
                        Wv_i [d_model, d_v] * h
                        Wo [d_v * h, d_model]

        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # contiguous(): after transpose() before view()
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Applied to each position separately and identically(use different parameters)

    FFN(x) = max(0, w1x + b1)w2 + b2
    """

    def __init__(self, d_model, argument=2, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * argument)
        self.w_2 = nn.Linear(d_model * argument, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Since Transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, must inject some information about the relative or absolute
    position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed

    Sinusoid version:
    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
        where pos is the position and i is the dimension
        That is, each dimension of the positional encoding corresponds to a sinusoid

    embeddings version:
    PE(pos) = embed(pos)
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # buffer: don't update
        self.register_buffer('pe', pe)

    def forward(self, x, reverse=False):
        t_num = x.size(1)
        if reverse:
            x = x + Variable(self.pe[:, :t_num], requires_grad=False)[:, [t_num - i - 1 for i in range(t_num)]]
        else:
            x = x + Variable(self.pe[:, :t_num], requires_grad=False)
        return self.dropout(x)
