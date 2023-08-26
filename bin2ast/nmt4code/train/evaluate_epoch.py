"""
evaluate on validation dataset
used to calculate validation loss after training every epoch on training set
return: average validation loss per sample
"""

import torch
from utils.tree import sequence_to_nary


def evaluate_epoch(model, iterator, criterion, model_type, output_lang, device, epoch_num):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch in iterator:
            src = batch[0].to(device)
            target = batch[1].to(device)
            loss = 0.0

            if model_type == "attention":
                src_lengths = batch[2]
                output = model(src, src_lengths, target)

            elif model_type == "transformer":
                src, target = src.T, target.T
                output, _ = model(src, target[:, :-1])

            elif model_type == "tree_decoder":
                src, targets = src.T, target.T
                target_trees = []
                for target in targets:
                    target_list = target.tolist()
                    # target sequence
                    target_str_all = [output_lang.index2token[key] for key in target_list]
                    # also remove the <pad> tokens as we will be working with one tree at a time for the tree decoder
                    target_str = [item for item in target_str_all if item != output_lang.pad_token]
                    # remove <sos> and <eos> token
                    target_str = target_str[1:-1]
                    # convert target sequence to target nary tree
                    target_tree_nary = sequence_to_nary(sequence=target_str,
                                                        vocab=output_lang.token2index,
                                                        eos_token=output_lang.eos_token)
                    # convert target nary tree to target binary tree
                    target_tree_binary = target_tree_nary.get_binary_tree_bfs()
                    target_trees.append(target_tree_binary)

                _, loss = model(src, target_trees, epoch_num)

            elif model_type == "gnn_decoder":
                src, targets = src.T, target.T
                target_trees = []
                for target in targets:
                    target_list = target.tolist()
                    # target sequence
                    target_str_all = [output_lang.index2token[key] for key in target_list]
                    # also remove the <pad> tokens as we will be working with one tree at a time for the tree decoder
                    target_str = [item for item in target_str_all if item != output_lang.pad_token]
                    # remove <sos> and <eos> token
                    target_str = target_str[1:-1]
                    # convert target sequence to target nary tree
                    target_tree_nary = sequence_to_nary(sequence=target_str,
                                                        vocab=output_lang.token2index,
                                                        eos_token=output_lang.eos_token)
                    target_trees.append(target_tree_nary)

                _, loss = model(src, target_trees, epoch_num)

            else:
                output = model(src, target)

            if model_type == "transformer":
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                target = target[:, 1:].contiguous().view(-1)
            elif model_type == "attention" or model_type == "basic":
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                target = target[1:].view(-1)

            if model_type != "tree_decoder" and model_type != "gnn_decoder":
                loss = criterion(output, target)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
