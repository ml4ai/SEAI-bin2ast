import torch
from torch import nn


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_lengths):
        # x: [src_len, batch_size]
        # src_lengths: [batch_size]
        embedded = self.dropout(self.embedding(x))
        # embedded: [src_len, batch_size, emb_dim]

        # for unpacked sequences
        # outputs, hidden = self.rnn(embedded)

        # for packing the padded sequence and then padding the outputs
        # put the lengths on the cpu
        src_lengths = src_lengths.to('cpu')
        # pack the padded embedded sequence using src_lengths
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        # packed_outputs is the packed sequence containing all hidden states
        # hidden is from the final non padded element in the batch
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs is now the padded sequence with pad tokens = 0

        # everything below should be similar to the old model
        # outputs: [src_len, batch_size, enc_hid_dim * n_directions]
        # hidden: [n_layers * n_directions, batch_size, hid_dim]
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        # hidden[-2, :, :] is the last of the forward RNN
        # hidden[-1, :, :] is the last of the backward RNN
        # initial decoder hidden is the final hidden state of the forward and backward
        # encoder RNN fed through the linear layer
        # we have n_directions = 2, so hidden.shape = [2, batch_size, hid_dim]
        # hidden[-2, :, :] = hidden[0, :, :] => index 0 has last output from forward RNN (h_t_forward)
        # hidden[-1, :, :] = hidden[1, :, :] => index 1 has last output from backward RNN (h_t_backward)
        # output.shape: [src_length, batch_size, [h_t_forward; h_t_backward]: t denotes the sequence length
        # so hidden[0, 0, :] = h_t_forward for last word in the first sequence = outputs[-1, 0, :512]
        # hidden[1, 0, :] = h_t_backward for the last word in the first sequence = outputs[0, 0, 512:]
        # they should be equal for all elements in the batch as well
        # assert torch.equal(hidden[-2, :, :], hidden[0, :, :])
        # assert torch.equal(hidden[-1, :, :], hidden[1, :, :])
        # assert torch.equal(hidden[0, :, :], outputs[-1, :, :self.enc_hid_dim])
        # assert torch.equal(hidden[1, :, :], outputs[0, :, self.enc_hid_dim:])
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden
