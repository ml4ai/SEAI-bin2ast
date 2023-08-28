import torch
from torch import nn


class TransformerEncoderGnnDecoder(nn.Module):
    def __init__(self, embedder, encoder, decoder, src_pad_idx):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    # def make_src_mask(self, source):
    #     # source: [batch_size, source_length]
    #     src_mask = (source != self.src_pad_idx).unsqueeze(dim=1).unsqueeze(dim=2)
    #     # src_mask: [batch_size, 1, 1, src_length]
    #     return src_mask

    def forward(self, src, target_trees, epoch_num):
        # src_mask = self.make_src_mask(src)
        # input_emb = self.encoder(src, src_mask)
        input_emb = self.embedder(src)
        input_emb = self.encoder(input_emb)
        return self.decoder(input_emb, target_trees, epoch_num)

    def predict(self, src):
        with torch.no_grad():
            input_emb = self.embedder(src)
            # src_mask = self.make_src_mask(src)
            # input_emb = self.encoder(src, src_mask)
            input_emb = self.encoder(input_emb)
            return self.decoder.predict(input_emb)
