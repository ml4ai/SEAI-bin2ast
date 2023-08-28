"""
Mutltihead attention and feedforward layer for transformer architecture
"""

import torch
from torch import nn
from torch.nn import functional as f


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, name, hid_dim, n_heads, dropout):
        super().__init__()
        self.name = name
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        # self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([self.head_dim])), persistent=False)

    def forward(self, query, key, value, mask=None):
        b_size = query.shape[0]
        # enc_self_attn: query_length = key_length = value_length = src_len
        # dec_self_attn: query_length = key_length = value_length = trg_len
        # dec_enc_attn: query_length = trg_len, key_length = src_len, value_length = src_len

        # query: [batch_size, query_length, hid_dim]
        # key: [batch_size, key_length, hid_dim]
        # value: [batch_size, value_length, hid_dim]

        # project into query, key and value space
        Q = self.fc_q(query)  # [batch_size, query_length, hid_dim]
        K = self.fc_k(key)  # [batch_size, key_length, hid_dim]
        V = self.fc_v(value)  # [batch_size, value_length, hid_dim]

        # break hid_dim into n_heads * head_dim, and swap n_heads with -1(seq_len)
        Q = Q.view(b_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(b_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(b_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q: [batch_size, n_heads, query_length, head_dim]
        # K: [batch_size, n_heads, key_length, head_dim]
        # V: [batch_size, n_heads, value_length, head_dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy: [batch_size, n_heads, query_length, key_length]
        # query_length = key_length = seq_len

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = f.softmax(energy, dim=-1)  # [batch_size, n_heads, query_length, key_length]
        # intrepretation: [for every sequence, for every head, for every word in the sequence,
        # we want [seq_len] weights that carry the attention weight to be given to that word

        x = torch.matmul(self.dropout(attention), V)  # [batch_size, n_heads, query_length, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_length, n_heads, head_dim]
        x = x.view(b_size, -1, self.hid_dim)  # [batch_size, query_length, hid_dim]
        x = self.fc_o(x)
        # x: [batch_size, query_length, hid_dim]
        # attention: [batch_size, n_heads, query_length, key_length]
        # enc_self_attn: query_length = key_length = src_len: [batch_size, n_heads, src_len, src_len]
        # dec_self_attn: query_length = key_length = trg_len: [batch_size, n_heads, trg_len, trg_len]
        # dec_enc_attn: query_length = trg_len, key_length = src_len: [batch_size, n_heads, trg_len, src_len]
        return x, attention


class PositionFeedForwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_length, hid_dim]
        x = self.dropout(torch.relu(self.fc1(x)))  # [batch_size, seq_length, pf_dim]
        x = self.fc2(x)  # [batch_size, seq_length, hid_dim]
        return x
