# Tree encoder and tree decoder
# implementaton of the paper: https://arxiv.org/pdf/1802.03691.pdf

from torch import nn


class TreeEncoderTreeDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_trees, target_trees, loss):
        self.encoder.forward(input_trees)
        return self.decoder.forward(input_trees, target_trees, loss)
