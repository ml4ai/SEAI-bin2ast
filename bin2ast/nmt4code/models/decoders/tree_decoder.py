# tree decoder for decoding trees using lstm:
# implementation of the paper: https://arxiv.org/pdf/1802.03691.pdf
# modified to work in input sequences also rather than only on the input tree

import torch
from torch import nn
from utils.tree import get_output_binary_tree_with_only_root
import random
import math


# Attention class to locate the source sub-tree and apply attention on input source tree
# used for: tree encoder and tree decoder
class InputTreeAttention(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # Weights matrices of size d * d (d is the embedding dimension)
        self.h_dim = h_dim
        self.w_0 = nn.Linear(h_dim, h_dim)
        self.w_1 = nn.Linear(h_dim, h_dim)
        self.w_2 = nn.Linear(h_dim, h_dim)

    # get the source tree
    def forward(self, tree, h_t):
        e_s = torch.zeros(1, self.h_dim)
        # dict of node ids to node
        node_dict = tree.get_node_dict()
        w_is = []
        h_is = []
        for _, node in node_dict.items():
            # compute the similarity: wi
            w_i = torch.exp(node.h @ self.w_0(h_t).T)
            w_is.append(w_i)
            h_is.append(node.h)

        # normalize w_is between 0 and 1 using softmax
        w_is = torch.softmax(torch.tensor(w_is), dim=-1)

        for index, h_i in enumerate(h_is):
            e_s += w_is[index] * h_i

        # compute e_t by combining W_1, W_2, e_s, and h_t and pass through activation function tanh
        e_t = torch.tanh(self.w_1(e_s) + self.w_2(h_t))
        return e_t


# Attention class to apply attention to input sequence
# used for: sequence encoder and tree decoder
# cross attention to input sequence for decoding a tree
class InputSequenceAttention(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # Weights matrices of size d * d (d is the embedding dimension)
        self.h_dim = h_dim
        self.w_0 = nn.Linear(h_dim, h_dim)
        self.w_1 = nn.Linear(h_dim, h_dim)
        self.w_2 = nn.Linear(h_dim, h_dim)

    def forward(self, input_emb, h_t):
        # note: this sequential version is extremely slow: comment out the sequential version
        # and replace it
        device = input_emb.device
        h_t = h_t.to(device)

        """
        # sequential version
        # e_s = torch.zeros(1, self.h_dim).to(device)
        # # dict of node ids to node
        # w_is = []
        # h_is = []
        # for emb in input_emb:
        #     # compute the similarity: wi
        #     w_i = torch.exp(emb @ self.w_0(h_t).T)
        #     w_is.append(w_i)
        #     h_is.append(emb)
        #
        # # normalize w_is between 0 and 1 using softmax
        # w_is = torch.softmax(torch.tensor(w_is), dim=-1).to(device)
        #
        # for index, h_i in enumerate(h_is):
        #     e_s += w_is[index] * h_i
        """

        # parallel version: input_emb: n * h, h_t: 1 * h
        h_t_prime = self.w_0(h_t).T  # h * 1
        w = torch.exp(torch.mm(input_emb, h_t_prime))  # n * 1
        w = torch.softmax(w, dim=0)  # n * 1
        e_s = torch.mm(input_emb.T, w)  # h * 1
        e_s = e_s.T  # 1 * h

        # compute e_t by combining W_1, W_2, e_s, and h_t and pass through activation function tanh
        e_t = torch.tanh(self.w_1(e_s) + self.w_2(h_t))
        return e_t


# Decoder generates the target tree starting from a single root node
class TreeDecoder(nn.Module):
    def __init__(self, h_dim, output_vocab, parent_feeding=False, input_type="sequence",
                 max_nodes=300, eos_token="<eos>", use_teacher_forcing=True):

        super().__init__()
        self.h_dim = h_dim
        self.output_vocab = output_vocab
        self.inv_output_vocab = {v: k for k, v in output_vocab.items()}
        # trainable matrix of vocab size of outputs and embedding dimension
        self.W_tt = nn.Linear(h_dim, len(self.output_vocab))
        self.B_t = nn.Embedding(len(self.output_vocab), h_dim)
        self.parent_feeding = parent_feeding
        self.use_teacher_forcing = use_teacher_forcing
        self.teacher_forcing_ratio = 0

        # attention mechanism
        self.input_type = input_type
        if self.input_type == "tree":
            self.attention = InputTreeAttention(h_dim)
        else:
            self.attention = InputSequenceAttention(h_dim)

        self.max_nodes = max_nodes
        self.eos_token = eos_token

        # lstm left and right
        if self.parent_feeding:
            # input size double of h_dim in case of parent feeding
            self.lstm_left = nn.LSTM(input_size=2 * h_dim, hidden_size=h_dim, num_layers=1, bias=False,
                                     batch_first=True, bidirectional=False)

            self.lstm_right = nn.LSTM(input_size=2 * h_dim, hidden_size=h_dim, num_layers=1, bias=False,
                                      batch_first=True, bidirectional=False)
        else:
            self.lstm_left = nn.LSTM(input_size=h_dim, hidden_size=h_dim, num_layers=1, bias=False,
                                     batch_first=True, bidirectional=False)

            self.lstm_right = nn.LSTM(input_size=h_dim, hidden_size=h_dim, num_layers=1, bias=False,
                                      batch_first=True, bidirectional=False)

    # generate target tree from source tree [or source sequence] depnding on context
    def forward(self, batch, target_trees, epoch_num):
        if self.use_teacher_forcing:
            self.teacher_forcing_ratio = round(math.exp(-epoch_num), 2)
        criterion = nn.CrossEntropyLoss()
        out_trees = []
        device = batch.device
        loss_list = list()
        for index, inp in enumerate(batch):
            target_tree = target_trees[index]
            # graph with only root node
            out_tree = get_output_binary_tree_with_only_root(output_vocab=self.output_vocab, root_id=0,
                                                             root_val="main", eos_token=self.eos_token)
            if self.input_type == "tree":
                tree = inp
                # if the input is a tree, then inp is tree
                # copy hidden states from the root of the encoded source tree
                out_tree.root.h = tree.root.h
                out_tree.root.c = tree.root.c
            else:
                # input is a sequnce: initialize the node h and c to zeros for now
                out_tree.root.h = torch.zeros(1, self.h_dim)
                out_tree.root.c = torch.zeros(1, self.h_dim)

            # guide the tree generation using target tree: else it won't converge
            node_list_target = [target_tree.root]
            node_list_gen = [out_tree.root]
            node_id = 0
            while node_list_target:
                # expand the first node in the queue
                current_node_target = node_list_target.pop(0)
                current_node = node_list_gen.pop(0)  # current node to expand in the out_tree
                e_t = self.attention(inp, current_node.h)
                current_node.e_t = e_t

                # feed it into softmax regression network to get our token
                # remove softmax as it will be calculated by the cross entropy loss
                # pred_prob = torch.softmax(self.W_tt(e_t), dim=-1)
                pred_prob = self.W_tt(e_t)
                # get label of a node with current node id(to expand) from tree (ground truth)
                # if node_id is not found: return label for "eos"
                true_label = torch.LongTensor([current_node_target.vocab_id]).to(device)
                # calculate loss
                # loss += criterion(pred_prob, true_label)
                temp_loss = criterion(pred_prob, true_label)
                loss_list.append(temp_loss)
                current_vocab_id = torch.argmax(pred_prob).item()
                current_token_val = self.inv_output_vocab[current_vocab_id]
                if current_node.vocab_id is None:
                    current_node.vocab_id = current_vocab_id
                    current_node.val = current_token_val

                # if the current target node has children: expand the current node of out_tree
                if current_node_target.left_child is not None or current_node_target.right_child is not None:
                    # calculate input vector to lstm
                    teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
                    if teacher_forcing:
                        input_lstm = self.B_t(torch.LongTensor([current_node_target.vocab_id]).to(device))
                    else:
                        input_lstm = self.B_t(torch.LongTensor([current_vocab_id]).to(device))

                    if self.parent_feeding:
                        input_lstm = torch.cat((input_lstm, current_node.e_t), dim=-1)

                    # add left child
                    node_id += 1
                    # left_node = out_tree.add_dummy_left_child(current_node)
                    left_node = out_tree.add_left_child(parent_node_id=current_node.node_id,
                                                        child_node_id=node_id,
                                                        child_val=None)
                    # calculate h and c for left node
                    _, (hl, cl) = self.lstm_left(input_lstm.unsqueeze(0),
                                                 (current_node.h.unsqueeze(0).to(device),
                                                  current_node.c.unsqueeze(0).to(device)))
                    left_node.h, left_node.c = hl.squeeze(0), cl.squeeze(0)
                    # add right child
                    node_id += 1
                    right_node = out_tree.add_right_child(parent_node_id=current_node.node_id,
                                                          child_node_id=node_id,
                                                          child_val=None)
                    _, (hr, cr) = self.lstm_right(input_lstm.unsqueeze(0),
                                                  (current_node.h.unsqueeze(0).to(device),
                                                   current_node.c.unsqueeze(0).to(device)))
                    right_node.h, right_node.c = hr.squeeze(0), cr.squeeze(0)

                    # add children to queue
                    node_list_gen.append(left_node)
                    node_list_gen.append(right_node)

                    # also add the target nodes
                    node_list_target.append(current_node_target.left_child)
                    node_list_target.append(current_node_target.right_child)
            out_trees.append(out_tree)
        loss = sum(loss_list) / len(loss_list)
        return out_trees, loss

    def predict(self, input_emb):
        out_trees = []
        device = input_emb.device
        for index, inp in enumerate(input_emb):
            # graph with only root node
            out_tree = get_output_binary_tree_with_only_root(output_vocab=self.output_vocab, root_id=0,
                                                             root_val="main", eos_token=self.eos_token)
            if self.input_type == "tree":
                tree = inp
                # if the input is a tree, then inp is tree
                # copy hidden states from the root of the encoded source tree
                out_tree.root.h = tree.root.h
                out_tree.root.c = tree.root.c
            else:
                # input is a sequnce: initialize the node h and c to zeros for now
                out_tree.root.h = torch.zeros(1, self.h_dim)
                out_tree.root.c = torch.zeros(1, self.h_dim)
            # initialize expanding node queue
            nodes_queue = [out_tree.root]
            # node_id for node to add: 0 for root
            node_id = 0
            # stop if there are no nodes left to expand: level expansion [root, left, right]
            while nodes_queue:
                # if the node_id crosses max_nodes [300] break out of generation
                if node_id == self.max_nodes:
                    break
                # expand the first node in the queue
                current_node = nodes_queue.pop(0)
                e_t = self.attention.forward(inp, current_node.h)
                current_node.e_t = e_t
                if current_node.is_root():
                    current_vocab_id = out_tree.root.vocab_id
                    current_token_val = out_tree.root.val
                else:
                    # feed it into softmax regression network to get our token
                    pred_prob = torch.softmax(self.W_tt(e_t), dim=-1)
                    current_vocab_id = torch.argmax(pred_prob).item()
                    current_token_val = self.inv_output_vocab[current_vocab_id]
                    if current_node.vocab_id is None:
                        current_node.vocab_id = current_vocab_id
                        current_node.val = current_token_val

                # if current_token_val isn't "<eoc>", make two children nodes [expand the current node]
                if current_token_val != "<eoc>":
                    # calculate input vector to lstm
                    input_lstm = self.B_t(torch.LongTensor([current_vocab_id]).to(device))
                    if self.parent_feeding:
                        input_lstm = torch.cat((input_lstm, current_node.e_t), dim=-1)

                    # add left child
                    node_id += 1
                    # left_node = out_tree.add_dummy_left_child(current_node)
                    left_node = out_tree.add_left_child(parent_node_id=current_node.node_id,
                                                        child_node_id=node_id,
                                                        child_val=None)
                    # calculate h and c for left node
                    _, (hl, cl) = self.lstm_left(input_lstm.unsqueeze(0),
                                                 (current_node.h.unsqueeze(0).to(device),
                                                  current_node.c.unsqueeze(0).to(device)))
                    left_node.h, left_node.c = hl.squeeze(0), cl.squeeze(0)
                    # add right child
                    node_id += 1
                    # right_node = out_tree.add_dummy_right_child(current_node)
                    right_node = out_tree.add_right_child(parent_node_id=current_node.node_id,
                                                          child_node_id=node_id,
                                                          child_val=None)
                    _, (hr, cr) = self.lstm_right(input_lstm.unsqueeze(0),
                                                  (current_node.h.unsqueeze(0).to(device),
                                                   current_node.c.unsqueeze(0).to(device)))
                    right_node.h, right_node.c = hr.squeeze(0), cr.squeeze(0)
                    # add children to queue
                    nodes_queue.append(left_node)
                    nodes_queue.append(right_node)
            out_trees.append(out_tree)

        return out_trees
