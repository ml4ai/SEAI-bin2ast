"""
optimized version of encoder to handle batch_size > 1
"""
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [src_len, batch_size]
        embedded = self.dropout(self.embedding(x))
        # embedded: [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [src_len, batch_size, hid_dim * num_directions]
        # hidden: [n_layers * n_directions, batch_size, hid_dim]
        # cell: [n_layers * n_directions, batch_size, hid_dim]
        return hidden, cell
