import torch
from torch import nn
from torch.nn import functional as f


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, dec_hid_dim]
        # encoder_outputs: [src_len, batch_size, enc_hid_dim * 2]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden states src_len times
        hidden = hidden.unsqueeze(dim=1).repeat(1, src_len, 1)  # [batch_size, src_len, dec_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, enc_hid_dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, dec_hid_dim]
        attention = self.v(energy).squeeze(dim=2)  # [batch_size, src_len]
        attention = attention.masked_fill(mask == 0, -1e10)
        return f.softmax(attention, dim=1)
