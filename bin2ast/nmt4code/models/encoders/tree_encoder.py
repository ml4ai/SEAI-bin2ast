# tree encoder for encoding input tree using tree lstm
# currently not used in our code base: our input is sequence instead of a tree
# but we can use it later: as we can augment the input sequence to a tree and
# encode it using tree lstm
# implementation of the papers:
# https://arxiv.org/pdf/1503.00075.pdf
# https://arxiv.org/pdf/1802.03691.pdf

import torch
from torch import nn


# tree LSTM cell for binary trees
class BinaryTreeLSTMCell(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.iou_x = nn.Linear(in_dim, h_dim * 3)  # i, o, u matrices for x (features)
        self.iou_hl = nn.Linear(h_dim, h_dim * 3)  # i, o, u matrices for left child
        self.iou_hr = nn.Linear(h_dim, h_dim * 3)  # i, o, u matrices for right child
        self.f_x = nn.Linear(in_dim, h_dim)  # forget for x

        # forget for hidden state
        self.f_h = nn.ModuleList([nn.Linear(h_dim, h_dim), nn.Linear(h_dim, h_dim),
                                  nn.Linear(h_dim, h_dim), nn.Linear(h_dim, h_dim)])

    # takes in input, cell states, and hidden states
    def forward(self, x, hl, hr, cl, cr):
        # i, o, u, gates
        iou = self.iou_x(x) + self.iou_hl(hl) + self.iou_hr(hr)

        # split
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)

        # apply activation functions
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # forget for left and right
        fl = torch.sigmoid(self.f_x(x) + self.f_h[0](hr) + self.f_h[1](hl))
        fr = torch.sigmoid(self.f_x(x) + self.f_h[2](hr) + self.f_h[3](hl))

        # calculate hidden state and cell state
        c = i * u + fl * cl + fr * cr
        h = o * torch.tanh(c)

        # return hidden state and cell state
        return h, c


class TreeEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, embedding_size):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.embed = nn.Embedding(in_dim, embedding_size)
        self.binary_cell = BinaryTreeLSTMCell(embedding_size, h_dim)

    # compute embeddings for source tree and subtrees
    def forward(self, input_trees):
        # iterate through each tree in batch
        for tree in input_trees:
            # get postorder of nodes for message passing from leafs to root
            node_stack = []

            def postorder(root):
                if root.left_child is not None:
                    postorder(root.left_child)
                if root.right_child is not None:
                    postorder(root.right_child)
                node_stack.append(root)

            postorder(tree.root)

            # message passing from leaf to root using binary tree lstm
            for node in node_stack:
                x = self.embed(torch.LongTensor([node.vocab_id]))
                if node.left_child is None or node.right_child is None:
                    hl, cl = torch.zeros(self.h_dim), torch.zeros(self.h_dim)
                    hr, cr = torch.zeros(self.h_dim), torch.zeros(self.h_dim)
                else:
                    hl, cl = node.left_child.h, node.left_child.c
                    hr, cr = node.right_child.h, node.right_child.c
                h, c = self.binary_cell(x, hl, hr, cl, cr)
                node.h = h
                node.c = c
