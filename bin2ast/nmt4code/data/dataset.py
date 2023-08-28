"""
Basic dataset classes for train, val and test data
"""

import random
import torch
from torch.utils.data import Dataset


class NMT4CodeDataset(Dataset):
    def __init__(self, input_data, target_data, data_augmentation=False, input_lang=None):
        super().__init__()
        self.data_augmentation = data_augmentation
        self.input_lang = input_lang
        self.input_data = input_data
        self.target_data = target_data

    def __getitem__(self, idx):
        inp = self.input_data[idx]
        # perform variable interchange as a data augmentation
        if self.data_augmentation:
            inp_list = inp.tolist()
            inp_str = [self.input_lang.index2token[key] for key in inp_list]
            unique_values = []
            # unique_address = []
            for token in inp_str:
                if '_v' in token:
                    if token not in unique_values:
                        unique_values.append(token)
                # if '_a' in token:
                #     if token not in unique_address:
                #         unique_address.append(token)

            # val_random = list(range(len(unique_values)))
            # random.shuffle(val_random)
            # add_random = list(range(len(unique_address)))
            # random.shuffle(add_random)
            # get random value to mask
            value_to_mask = random.choice(unique_values)
            # target_values = ['_v' + str(key) for key in val_random]
            # target_address = ['_a' + str(key) for key in add_random]
            # val_dict = dict(zip(unique_values, target_values))
            # add_dict = dict(zip(unique_address, target_address))
            new_inp_str = []
            for token in inp_str:
                if token == value_to_mask:
                    new_inp_str.append('<unk>')
                # elif token in add_dict:
                #     new_inp_str.append(add_dict[token])
                else:
                    new_inp_str.append(token)
            new_inp = [self.input_lang.token2index[key] for key in new_inp_str]
            inp = torch.LongTensor(new_inp)
        return [inp, self.target_data[idx]]

    def __len__(self):
        return len(self.input_data)
