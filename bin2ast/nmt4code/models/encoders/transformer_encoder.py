"""
Encoder architecture for transformer
"""

import torch
from torch import nn
from models.transformer.transformer import MultiHeadAttentionLayer, PositionFeedForwardLayer


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer("enc_self_attn", hid_dim, n_heads, dropout)
        self.pos_ff = PositionFeedForwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, source_mask):
        # source: [batch_size, src_len, hid_dim]
        # source_mask: [batch_size, 1, 1, source_length]
        _src, _ = self.self_attention(source, source, source, source_mask)
        # _src: [batch_size, src_len, hid_dim]

        # dropout, residual connection and layer norm
        source = self.self_attn_layer_norm(source + self.dropout(_src))  # [batch_size, src_len, hid_dim]
        # position wise feedforward
        _src = self.pos_ff(source)  # [batch_size, src_len, hid_dim]
        # dropout, residual and layer norm
        source = self.ff_layer_norm(source + self.dropout(_src))  # [batch_size, src_len, hid_dim]
        return source


class TransformerEncoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length=100):
        super().__init__()
        self.tok_embedding = nn.Embedding(inp_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([hid_dim])), persistent=False)

    def forward(self, source, source_mask):
        # source: [batch_size, source_length]
        # source_mask: [batch_size, 1, 1, source_length]
        device = source.device
        b_size, source_length = source.shape
        pos = torch.arange(0, source_length).unsqueeze(dim=0).repeat(b_size, 1).to(device)
        # pos: [batch_size, source_length]
        pos_emb = self.pos_embedding(pos)
        tok_emb = self.tok_embedding(source)
        src_emb = self.dropout((tok_emb * self.scale) + pos_emb)
        # source: [batch_size, soruce_length, hid_dim]
        for layer in self.layers:
            src_emb = layer(src_emb, source_mask)

        return src_emb
