"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    hidden: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    embedding: int = 128 # Factorized embedding parameterization

    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 768//64 # Numher of Heads in Multi-Headed Attention Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden))
        self.beta  = nn.Parameter(torch.zeros(cfg.hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        # Original BERT Embedding
        # self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden) # token embedding

        # factorized embedding
        self.tok_embed1 = nn.Embedding(cfg.vocab_size, cfg.embedding)
        self.tok_embed2 = nn.Linear(cfg.embedding, cfg.hidden)

        self.pos_embed = nn.Embedding(cfg.max_len, cfg.hidden) # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.hidden) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        # factorized embedding
        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        e = e + self.pos_embed(pos) + self.seg_embed(seg)
        #return self.drop(self.norm(e))
        return self.norm(e)

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden) # Wq
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden) # Wk
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden) # Wv
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # n_heads는 서로 다른 방법으로 attention을 학습하는 것들을 다시 모으는것.
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1)) # scale dot-product
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h
    # h=6라 하고, n_head = 3이면, 6차원짜리 벡터가 seq 갯수만큼 존재. 이 때 Wq, Wk, Wv를 곱해서 q,k,v 가 각각 기존 x와 같은 size를 가지도록 함(B, s, 6)
    # 이걸 이제 (B, s, 3, 2)로 쪼개고, merge 전까지 계산된 h는 (B, s, 3, 2)짜리 tensor인데  이를 (B, s, 6)짜리로 만듬. head마다의 결과를 concat함.


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


# class Block(nn.Module):
#     """ Transformer Block """
#     def __init__(self, cfg):
#         super().__init__()
#         self.attn = MultiHeadedSelfAttention(cfg)
#         self.proj = nn.Linear(cfg.hidden, cfg.hidden)
#         self.norm1 = LayerNorm(cfg)
#         self.pwff = PositionWiseFeedForward(cfg)
#         self.norm2 = LayerNorm(cfg)
#         self.drop = nn.Dropout(cfg.p_drop_hidden)
#
#     def forward(self, x, mask):
#         h = self.attn(x, mask)
#         h = self.norm1(x + self.drop(self.proj(h)))
#         h = self.norm2(h + self.drop(self.pwff(h)))
#         return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden) # head의 결과를 1번 proj한다음에 residual 연산을 수행.
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg) # 이제 seq마다 따로 FC를 연산을 수행해서 다시한번 마지막에 residual 연산을 수행.
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)

        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h, mask)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h)) # [batch_size, seq_len, dimension]

        return h

