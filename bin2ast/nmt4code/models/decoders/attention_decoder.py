import torch
from torch import nn


class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim * 2 + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim * 2 + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, hidden, encoder_outputs, mask):
        # inp: [batch_size]
        # hidden: [batch_size, dec_hid_dim]
        # encoder_outputs: [src_len, batch_size, enc_hid_dim * 2]
        inp = inp.unsqueeze(dim=0)  # [1, batch_size]
        # seq_length = 1 for decoder [we are decoding one word at a time]
        embedded = self.dropout(self.embedding(inp))  # embedded: [1, batch_size, emb_dim]
        a = self.attention(hidden, encoder_outputs, mask)  # [batch_size, src_len]
        a = a.unsqueeze(dim=1)  # [batch_size, 1, src_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, enc_hid_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, enc_hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)  # [1, batch_size, enc_hid_dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, enc_hid_dim * 2 + emb_dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(dim=0))
        # output: [seq_len, batch_size, dec_hid_dim * n_directions]
        # hidden: [n_layers * n_directions, batch_size, dec_hid_dim]
        # we have seq_len = 1, n_layers = 1 and n_directions = 1 for decoder, so
        # output: [1, batch_size, dec_hid_dim]
        # hidden: [1, batch_size, dec_hid_dim]
        # assert torch.equal(hidden, output)
        embedded = embedded.squeeze(dim=0)
        output = output.squeeze(dim=0)
        weighted = weighted.squeeze(dim=0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch_size, output_dim]
        return prediction, hidden.squeeze(dim=0)
