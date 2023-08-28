"""
initial simple lstm encoder module
works for a single sample (batch_size = 1) and one word at a time
now a legacy code: only used for testing
"""

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, inp, hidden):
        embedding = self.embedding(inp).view(1, 1, -1)
        output = embedding
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self, device):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))
