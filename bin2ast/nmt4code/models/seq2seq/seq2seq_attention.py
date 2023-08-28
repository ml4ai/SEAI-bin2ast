"""
optimized version of seq2seq to handle batch_size > 1
puts encoder and decoder into a single module: creates a seq2seq abstraction
using encoder and decoder
"""

import torch
from torch import nn
import torch.nn.functional as f


class SequenceToSequenceWithAttention(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def create_mask(self, source):
        mask = (source != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, inp, src_lengths, target):
        # inp: [src_len, batch_size]
        # target: [target_len, batch_size]
        b_size = target.shape[1]
        trg_len = target.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, b_size, trg_vocab_size).to(self.device)

        # encoder_outputs: all hidden states of the input sequence: forward and backward
        # hidden is the final forward and backward hidden states, passed through the linear layer
        encoder_outputs, hidden = self.encoder(inp, src_lengths)

        # first input to decoder is the <sos> token
        decoder_input = target[0, :]
        # create mask
        mask = self.create_mask(inp)  # [batch_size, src_len]
        for t in range(1, trg_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            outputs[t] = output
            decoder_input = output.argmax(dim=1)
        return outputs

    def predict(self, inp_tensor, target_sos_token, target_eos_token, max_length=500):
        """
        predict output sequence given input sequence only
        :param inp_tensor: single input token sequence
        :param target_sos_token: sos token index for target language
        :param target_eos_token: eos token index for target language
        :param max_length: max length the model supports
        :return: target sequence
        """
        encoder_outputs, hidden = self.encoder(inp_tensor, torch.tensor([inp_tensor.shape[0]]))
        # first input to decoder is the <sos> token
        preds = [target_sos_token]
        # create mask
        mask = self.create_mask(inp_tensor)  # [batch_size, src_len]
        for t in range(max_length):
            output, hidden = self.decoder(torch.tensor([preds[-1]]).to(self.device),
                                          hidden, encoder_outputs, mask)
            pred = output.argmax(dim=1)
            preds.append(pred.item())
            if pred == target_eos_token:
                break
        return preds

    def beam_decode(self, inp, target, beam_width=5):
        # inp: [src_len, batch_size]
        # target: [target_len, batch_size]
        trg_len = target.shape[0]

        # last hidden state of the encoder used as initial hidden state for the decoder
        hidden, cell = self.encoder(inp)

        # first input to decoder is the <sos> token
        decoder_input = target[0, :]
        # dictionary to store the current possible sequences with log probabilities
        temp_log_prob = {}
        # dictionary to store the current possible sequences with their
        # corresponding hidden and cell states
        temp_hidden_cell = {}
        # dictionary that stores the best sequences from log_prob
        best = {}
        # dictionary that stores the hidden and cell state for each sequence in the best dict
        best_hidden_cell = {}

        with torch.no_grad():
            for t in range(1, trg_len):
                if t == 1:
                    output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                    log_softmax = f.log_softmax(output, dim=1)
                    topv, topi = torch.topk(log_softmax, k=beam_width, dim=1)
                    for idx in range(beam_width):
                        key = str(decoder_input.item()) + "," + str(topi[0][idx].item())
                        best[key] = topv[0][idx].item()
                        best_hidden_cell[key] = (hidden, cell)
                else:
                    for key, val in best.items():
                        output, hidden, cell = self.decoder(torch.tensor([int(key[-1])]),
                                                            best_hidden_cell[key][0],
                                                            best_hidden_cell[key][1])
                        log_softmax = f.log_softmax(output, dim=1)
                        topv, topi = torch.topk(log_softmax, k=beam_width, dim=1)
                        for idx in range(beam_width):
                            current_key = topi[0][idx].item()
                            new_key = key + ',' + str(current_key)
                            temp_log_prob[new_key] = val + topv[0][idx].item()
                            temp_hidden_cell[new_key] = (hidden, cell)

                    # select best values based on log_prob
                    best = {k: v for k, v in sorted(temp_log_prob.items(),
                                                    key=lambda x: x[1],
                                                    reverse=True)[:beam_width]}
                    for k, _ in best.items():
                        best_hidden_cell[k] = temp_hidden_cell[k]
                    temp_log_prob = {}
                    temp_hidden_cell = {}
        result = []
        for key, _ in best.items():
            result.append(key)
        return result
