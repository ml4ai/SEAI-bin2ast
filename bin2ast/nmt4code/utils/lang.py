"""
driver code that keeps the statistics of input language and output language
"""

import torch


class InputLanguage:
    def __init__(self, name):
        self.name = name
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.token2index = {self.pad_token: 0, self.unk_token: 1,
                            self.sos_token: 2, self.eos_token: 3}
        self.index2token = {idx: tok for tok, idx in self.token2index.items()}
        self.tokencount = {}
        self.n_tokens = 4  # <pad>, <unk>, <sos>, <eos>

    def add_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.tokencount[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.tokencount[token] += 1

    def add_tokens(self, tokens_list):
        # add <sos> token to start of sequence
        return_list = [self.token2index[self.sos_token]]
        for item in tokens_list:
            token = item.strip()
            self.add_token(token)
            return_list.append(self.token2index[token])

        # add <eos> token to end of sequence
        return_list.append(self.token2index[self.eos_token])
        return torch.tensor(return_list, dtype=torch.long)

    def tensor_from_sequence(self, tokens_list):
        # follow add_tokens protocol: for val/test set
        # also return number of 'unk' tokens for debugging
        count_unk = 0
        unk_index = self.token2index[self.unk_token]
        # add <sos> token to start of sequence
        indexes = [self.token2index[self.sos_token]]

        for item in tokens_list:
            token = item.strip()
            index = self.token2index.get(token, unk_index)
            if index == unk_index:
                count_unk += 1
            indexes.append(index)

        # add <eos> token to end of sequence
        indexes.append(self.token2index[self.eos_token])

        return torch.tensor(indexes, dtype=torch.long), count_unk


class OutputLanguage:
    def __init__(self, name):
        self.name = name
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.token2index = {self.pad_token: 0, self.unk_token: 1,
                            self.sos_token: 2, self.eos_token: 3}
        self.index2token = {idx: tok for tok, idx in self.token2index.items()}
        self.tokencount = {}
        self.n_tokens = 4  # <pad>, <unk>, <sos>, <eos>

    def add_eoc(self):
        """
        add eoc token: needed for tree / graph generative models
        """
        new_id = len(self.token2index)
        self.index2token.update({new_id: "<eoc>"})
        self.token2index.update({"<eoc>": new_id})
        self.n_tokens += 1

    def add_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.tokencount[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.tokencount[token] += 1

    def add_tokens(self, tokens_list):
        # add <sos> token to start of sequence
        return_list = [self.token2index[self.sos_token]]
        for item in tokens_list:
            token = item.strip()
            self.add_token(token)
            return_list.append(self.token2index[token])

        # add <eos> token to end of sequence
        return_list.append(self.token2index[self.eos_token])
        return torch.tensor(return_list, dtype=torch.long)

    def tensor_from_sequence(self, tokens_list):
        # follow add_tokens protocol: for val/test set
        # also return count_unk for debugging: unknown tokens
        count_unk = 0
        unk_index = self.token2index[self.unk_token]
        # add <sos> token to start of sequence
        indexes = [self.token2index[self.sos_token]]
        # tokens_list = sequence.split()
        for item in tokens_list:
            token = item.strip()
            index = self.token2index.get(token, unk_index)
            if index == count_unk:
                count_unk += 1
            indexes.append(index)
        # add <eos> token to end of sequence
        indexes.append(self.token2index[self.eos_token])
        return torch.tensor(indexes, dtype=torch.long), count_unk
