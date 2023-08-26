"""
sequence to sequence network with transformer
"""
import torch
from torch import nn


class SequenceToSequenceWithTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, source):
        # source: [batch_size, source_length]
        src_mask = (source != self.src_pad_idx).unsqueeze(dim=1).unsqueeze(dim=2)
        # src_mask: [batch_size, 1, 1, src_length]
        return src_mask

    def make_trg_mask(self, target):
        # target: [batch_size, target_length]
        device = target.device
        trg_pad_mask = (target != self.trg_pad_idx).unsqueeze(dim=1).unsqueeze(dim=2)
        # trg_pad_mask: [batch_size, 1, 1, trg_len]
        trg_len = target.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
        # trg_sub_mask: [trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, trg_len, trg_len]
        return trg_mask

    def forward(self, source, target):
        # source: [batch_size, src_len]
        # target: [batch_size, trg_len]
        src_mask = self.make_src_mask(source)  # [batch_size, 1, 1, src_len]
        trg_mask = self.make_trg_mask(target)  # [batch_size, 1, trg_len, trg_len]
        enc_src = self.encoder(source, src_mask)  # [batch_size, src_len, hid_dim]
        output, attention = self.decoder(target, enc_src, trg_mask, src_mask)
        # output: [batch_size, trg_len, output_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]
        return output, attention

    def predict(self, inp_tensor, target_sos_token, target_eos_token, max_length=500,
                return_attention_map=False):
        """
        predict output sequence given input sequence only
        :param inp_tensor: single input token sequence
        :param target_sos_token: sos token index for target language
        :param target_eos_token: eos token index for target language
        :param max_length: max length the model supports
        :param return_attention_map: optionally return an attention map
        :return: target sequence
        """
        inp_mask = self.make_src_mask(inp_tensor)
        preds = [target_sos_token]
        device = inp_tensor.device
        with torch.no_grad():
            enc_src = self.encoder(inp_tensor, inp_mask)

        for i in range(max_length):
            target = torch.tensor(preds).unsqueeze(dim=0).to(device)
            trg_mask = self.make_trg_mask(target)
            with torch.no_grad():
                output, attention = self.decoder(target, enc_src, trg_mask, inp_mask)
            pred = output.argmax(dim=2)[:, -1].item()
            preds.append(pred)
            if pred == target_eos_token:
                break
        if return_attention_map:
            return preds, attention
        return preds
