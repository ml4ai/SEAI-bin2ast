"""
data loader for optimized code: batch_size > 1
uses collate_fn to create custom batch generator and pads the variable length
sequences with pad_token
"""

from data.dataset import NMT4CodeDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data.distributed import DistributedSampler


def pad_collate(batch):
    src = []
    trg = []
    src_lengths = []
    for source, target in batch:
        src.append(source)
        trg.append(target)
        src_lengths.append(len(source))
    padded_src = pad_sequence(src, batch_first=False, padding_value=0)
    padded_trg = pad_sequence(trg, batch_first=False, padding_value=0)
    return padded_src, padded_trg, torch.tensor(src_lengths)


def get_loaders(train_input_data, train_target_data, val_input_data, val_target_data,
                test_input_data, test_target_data, input_lang, data_augmentation=False,
                batch_size=128, is_distributed=False):
    # create the respective datasets
    train_dataset = NMT4CodeDataset(train_input_data,
                                    train_target_data,
                                    data_augmentation=data_augmentation,
                                    input_lang=input_lang)

    val_dataset = NMT4CodeDataset(val_input_data,
                                  val_target_data,
                                  data_augmentation=False,
                                  input_lang=None)

    test_dataset = NMT4CodeDataset(test_input_data,
                                   test_target_data,
                                   data_augmentation=False,
                                   input_lang=None)

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False if train_sampler else True,
                              pin_memory=False,
                              collate_fn=pad_collate,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=pad_collate)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=pad_collate)

    return train_loader, val_loader, test_loader
