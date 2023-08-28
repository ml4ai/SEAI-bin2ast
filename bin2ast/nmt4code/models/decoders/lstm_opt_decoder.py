"""
optimized version of the lstm decoder to handle batch_size > 1
"""
from torch import nn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers,
                           dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, hidden, cell):
        # inp: [batch_size]
        # hidden: [n_layers * n_directions, batch_size, hid_dim]
        # cell: [n_layers * n_directions, batch_size, hid_dim]
        inp = inp.unsqueeze(dim=0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(inp))
        # embedded: [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [seq_len, batch_size, hid_dim * n_directions]
        prediction = self.fc_out(output.squeeze(dim=0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell
