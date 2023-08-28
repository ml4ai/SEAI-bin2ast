"""
train for one epoch: used in train_opt.py (train optimized) to train each epoch on
the training dataset
performs backpropagation and weight update and then
returns: average training loss per sample
"""

import torch
from utils.tree import sequence_to_nary


def train_epoch(model, loader, optimizer, criterion, clip, model_type,
                output_lang, rank,
                epoch_num, is_distributed=True,
                print_loss_minibatch=True):
    model.train()
    epoch_loss = 0.0
    batch_counter = 0
    for batch in loader:
        src = batch[0].to(rank)
        batch_size = src.T.shape[0]
        target = batch[1].to(rank)
        optimizer.zero_grad()
        output = None
        loss = None
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
                # as tree decoder works with binary trees
                target_tree_binary = target_tree_nary.get_binary_tree_bfs()
                target_trees.append(target_tree_binary)

            out_trees, loss = model(src, target_trees, epoch_num)

        elif model_type == "gnn_decoder":
            src, targets = src.T, target.T
            target_graphs = []
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

                # gnn_decoder works with nary trees
                target_graphs.append(target_tree_nary)

            out_graphs, loss = model(src, target_graphs, epoch_num)
        else:
            output = model(src, target)

        if model_type == "transformer":
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = target[:, 1:].contiguous().view(-1)

        elif model_type == "attention" or model_type == "basic":
            # target: [target_len, batch_size]
            # output: [target_len, batch_size, output_dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = target[1:].view(-1)

        if model_type != "tree_decoder" and model_type != "gnn_decoder":
            loss = criterion(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        if print_loss_minibatch:
            minibatch_loss = round((loss.item() / batch_size), 2)
            if is_distributed:
                if rank == 0:
                    print(f"epoch: {epoch_num}, batch: {batch_counter}, loss: {minibatch_loss}")
            else:
                print(f"epoch: {epoch_num}, batch: {batch_counter}, loss: {minibatch_loss}")

        batch_counter += 1

    return epoch_loss / len(loader)
