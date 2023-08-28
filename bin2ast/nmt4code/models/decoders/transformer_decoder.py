"""
Decoder architecture for transformer
"""

import torch
from torch import nn
from models.transformer.transformer import MultiHeadAttentionLayer, PositionFeedForwardLayer


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer("dec_self_attn", hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer("dec_enc_attn", hid_dim, n_heads, dropout)
        self.pos_ff = PositionFeedForwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, enc_src, trg_mask, src_mask):
        # target: [batch_size, trg_len, hid_dim]
        # enc_src: [batch_size, src_length, hid_dim]
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        # src_mask: [batch_size, 1, 1, src_length]
        # self attention
        _trg, _ = self.self_attention(target, target, target, trg_mask)  # [batch_size, trg_len, hid_dim]
        # dropout, residual connection and layer norm
        target = self.self_attn_layer_norm(target + self.dropout(_trg))  # [batch_size, trg_len, hid_dim]
        # encoder attention
        _trg, attention = self.encoder_attention(target, enc_src, enc_src, src_mask)
        # _trg: [batch_size, trg_len, hid_dim]: can be though as target word embedding
        # that has information from it's preceding words in the target side, as well as the
        # contextualized embeddings (enc_src) from source sentence
        # attention: [batch_size, n_heads, trg_len, src_len]: importance to the src_words while generating
        # each of the target word

        # dropout, residual connection and layer norm
        target = self.enc_attn_layer_norm(target + self.dropout(_trg))  # [batch_size, trg_len, hid_dim]
        # positionwise feedforward
        _trg = self.pos_ff(target)  # [batch_size, trg_len, hid_dim]
        # dropout, residual connection and layer norm
        target = self.ff_layer_norm(target + self.dropout(_trg))
        # target: [batch_size, trg_len, hid_dim]
        # attention: [batch_size, n_heads, trg_len, src_length]
        return target, attention


class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length=100):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([hid_dim])), persistent=False)

    def forward(self, target, enc_src, trg_mask, src_mask):
        # target: [batch_size, target_length]
        # enc_src: [batch_size, src_length, hid_dim]
        # trg_mask: [batch_size, 1, target_length, target_length]
        # src_mask: [batch_size, 1, 1, src_length]
        b_size, trg_len = target.shape
        device = target.device
        pos = torch.arange(0, trg_len).unsqueeze(dim=0).repeat(b_size, 1).to(device)
        # pos: [batch_size, target_length]
        pos_emb = self.pos_embedding(pos)
        tok_emb = self.tok_embedding(target)
        target_emb = self.dropout((tok_emb * self.scale) + pos_emb)
        # target: [batch_size, target_length, hid_dim]
        attention = None
        for layer in self.layers:
            target_emb, attention = layer(target_emb, enc_src, trg_mask, src_mask)

        # target: [batch_size, target_length, hid_dim]
        # attention: [batch_size, n_heads, target_length, src_length]
        # This captures the importance of input words while predicting the target word in the decoder

        output = self.fc_out(target_emb)  # [batch_size, target_length, output_dim]
        return output, attention
