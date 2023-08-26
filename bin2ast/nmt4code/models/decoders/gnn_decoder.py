import dgl
import torch
from torch import nn
import math
import torch.nn.functional as F
from utils.tree import get_output_nary_tree_with_only_root
import random


# todo: add attention to partial graph
# todo: add multiple layers of attentions like in transformer: currently a single layer


class GCN(nn.Module):
    def __init__(self, h_feats, num_propagation=2):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(h_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, h_feats)
        # self.conv_modules = nn.ModuleList([dgl.nn.GraphConv(h_feats, h_feats) for _ in range(num_propagation)])

    def forward(self, g, h_feat):
        h = self.conv1(g, h_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
        # for module in self.conv_modules:
        #     h_feat = F.relu(module(g, h_feat))
        # return h_feat


class PositionalEncoding(nn.Module):

    def __init__(self, h_dim, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, h_dim, 2) * (-math.log(10000.0) / h_dim))
        pe = torch.zeros(max_len, 1, h_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, index):
        """
        Args:
            index: child index for positional encoding
        """
        return self.pe[index]


class CrossAttention(nn.Module):
    """
    Attention module that will predict the h vector for current child given the parent_h,
    input_emb, and child_index: cross attention module
    """

    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.w_k = nn.Linear(2 * h_dim, h_dim)
        self.w_q = nn.Linear(h_dim, h_dim)
        self.w_v = nn.Linear(h_dim, h_dim)
        self.pos_enc = PositionalEncoding(h_dim)
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([self.h_dim])), persistent=False)

    def forward(self, input_emb, current_h, child_index):
        # sine and cosine positional encoding for child_index
        # works for only one batch
        # right now attends over just the input sequnce but not over the generated graph
        # later: modify it to attend over the generated partial graph as well
        # get the pos_enc for 0 index
        pos_enc = self.pos_enc(child_index)[0]
        # current_h += pos_enc
        current_h = torch.cat((current_h, pos_enc), dim=0)
        key = self.w_k(current_h)  # h_dim
        query = self.w_q(input_emb)  # n * h_dim
        values = self.w_v(input_emb)  # n * h_dim

        """sequential version:
        device = input_emb.device
        alphas = []
        for vec in query:
            alpha = torch.dot(vec, key) / math.sqrt(self.h_dim)
            alphas.append(alpha)

        out = torch.zeros(self.h_dim).to(device)
        for index, alpha in enumerate(alphas):
            out += alpha * values[index]

        return out
        """

        alphas = torch.mm(query, key.reshape(-1, 1)) / self.scale  # n * 1
        alphas = torch.softmax(alphas, dim=0)  # n * 1
        out = torch.mm(values.T, alphas)[:, 0]  # h_dim
        return out


class GnnDecoder(nn.Module):
    """
    GNNDecoder that takes the input embeddings and generates a graph
    """

    def __init__(self, num_propagation, h_dim, out_vocab, eos_token="<eos>",
                 max_nodes=100, use_teacher_forcing=True):
        super().__init__()
        self.num_propagation = num_propagation
        self.h_dim = h_dim
        out_dim = len(out_vocab)
        self.out_dim = out_dim
        self.w_out = nn.Linear(h_dim, out_dim)
        self.emb = nn.Embedding(out_dim, h_dim)
        self.out_vocab = out_vocab
        self.out_vocab_inv = {v: k for k, v in out_vocab.items()}
        self.cross_attention = CrossAttention(h_dim=h_dim)
        self.message_passing_network = GCN(h_dim, num_propagation)
        self.max_nodes = max_nodes
        self.use_teacher_forcing = use_teacher_forcing
        self.teacher_forcing_ratio = 0
        self.eos_token = eos_token

    def forward(self, input_emb, target_graphs, epoch_num):
        if self.use_teacher_forcing:
            self.teacher_forcing_ratio = round(math.exp(-epoch_num), 2)
        criterion = nn.CrossEntropyLoss()
        out_graphs = []
        device = input_emb.device
        loss_list = list()
        for index, inp in enumerate(input_emb):
            target_graph = target_graphs[index]

            # out_graph: our instance of graph: just for bookkeeping
            out_graph = get_output_nary_tree_with_only_root(output_vocab=self.out_vocab, root_id=0,
                                                            root_val="main", eos_token=self.eos_token)
            # dgl_graph: does actual storage and message passing
            # initialize initial graph with root node
            dgl_graph = dgl.graph(data=([], []), num_nodes=1).to(device)
            # initialize hidden states for each node
            dgl_graph.ndata['h'] = torch.zeros(1, self.h_dim).to(device)
            node_list_target = [target_graph.root]
            # initialize node id in bfs order
            node_id = 0
            # bookkeeping
            out_graph_node_list = [out_graph.root]

            while node_list_target:
                current_node_target = node_list_target.pop(0)
                out_graph_current_node = out_graph_node_list.pop(0)
                current_node_id = out_graph_current_node.node_id
                # current_node_h = dgl_graph.ndata['h'][current_node_id]

                # if we see "eoc" token: do not expand
                # if current_node_target.val == "<eoc>":
                #     continue

                # predict the childs of the current nodes and add them to the nodes list
                child_index = 0
                num_children = len(current_node_target.children)

                while child_index < num_children:
                    teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
                    if teacher_forcing:
                        parent_h = self.emb(torch.LongTensor([current_node_target.vocab_id]).to(device))[0]
                    else:
                        parent_h = self.emb(torch.LongTensor([out_graph_current_node.vocab_id]).to(device))[0]

                    child_h = F.relu(self.cross_attention(inp, parent_h, child_index))
                    out_prob = self.w_out(child_h).reshape(1, -1)
                    current_target_child = current_node_target.children[child_index]
                    child_vocab_id = current_target_child.vocab_id
                    true_label = torch.LongTensor([child_vocab_id]).to(device)
                    # loss += criterion(out_prob, true_label)
                    temp_loss = criterion(out_prob, true_label)
                    loss_list.append(temp_loss)

                    child_index += 1
                    node_id += 1
                    # update lists
                    node_list_target.append(current_target_child)

                    # bookkeeping
                    predicted_token_vocab_id = torch.argmax(out_prob.detach()).item()
                    predicted_token_name = self.out_vocab_inv[predicted_token_vocab_id]
                    child = out_graph.add_child(parent_node_id=out_graph_current_node.node_id,
                                                child_node_id=node_id,
                                                child_val=predicted_token_name,
                                                return_child=True)
                    out_graph_node_list.append(child)

                    # add given node to the graph
                    dgl_graph.add_nodes(1)
                    dgl_graph.ndata['h'][-1, :] = child_h
                    # add edge from current parent to the current child node
                    dgl_graph.add_edges([current_node_id], [node_id])
                    # if we want to add the reverse edge
                    dgl_graph.add_edges([node_id], [current_node_id])
                    # perform message passing after adding one child
                    # and update the ndata hidden tensor
                    dgl_graph.ndata['h'] = self.message_passing_network(dgl_graph, dgl_graph.ndata['h'])

            out_graphs.append(out_graph)
        loss = sum(loss_list) / len(loss_list)
        return out_graphs, loss

    def predict(self, input_emb):
        out_graphs = []
        # device = input_emb.device
        for inp in input_emb:
            graph = dgl.graph(data=([], []), num_nodes=1)
            graph.ndata['h'] = torch.randn(1, self.h_dim)
            # print decoded list for debugging: bfs order
            decoded_list = ['root']
            # nodeid list and vocabid list to expand: two list with corresponding pair
            nodeid_list = [0]
            vocabid_list = [0]
            # node id tracker to add to the given graph
            nodeid_counter = 0
            # set max_nodes: 500 for now
            # max_nodes = 50
            # ground truth nodeid expansion order
            true_node_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            true_vocab_order = [0, 7, 7, 1, 3, 4, 1, 5, 6, 1, 1, 1, 1, 2]
            # without teacher forcing
            # while nodeid_list and nodeid_counter < max_nodes and current_node_name is not "eos":
            # with teacher forcing
            while true_node_order:
                # run this loop untill we see <eos>
                # current node id is the node id of the node that is being expanded (parent)

                # no teacher forcing
                # current_node_id = nodeid_list.pop(0)
                # current_vocab_id = vocabid_list.pop(0)

                # teacher forcing
                current_node_id = true_node_order.pop(0)
                current_vocab_id = true_vocab_order.pop(0)

                current_node_name = self.out_vocab_inv[current_vocab_id]
                current_node_h = graph.ndata['h'][current_node_id]
                # if we see "eoc" token: do not expand
                if current_node_name == "eoc":
                    continue
                # else: predict the childs of the current nodes and add them to the nodes list
                child_index = 0
                # set max_number of childs : 10 for now
                # max_children = 5
                num_children = 5
                # print(f"currently expanding node: {current_node_name}")
                # while predicted_token_name not in ["eoc", "eos"] and child_index < max_children:
                while child_index < num_children:
                    # generate childs untill we see <eoc> child or "eos" child
                    child_h = self.cross_attention(inp, current_node_h, child_index)
                    out_prob = self.w_out(child_h).reshape(1, -1)
                    predicted_token_vocab_id = torch.argmax(out_prob.detach()).item()
                    predicted_token_name = self.out_vocab_inv[predicted_token_vocab_id]
                    decoded_list.append(predicted_token_name)
                    # print(f"{child_index}th child of {current_node_name}: {predicted_token_name}")
                    child_index += 1
                    nodeid_counter += 1
                    # update nodeid_list and vocabid_list
                    nodeid_list.append(nodeid_counter)
                    vocabid_list.append(predicted_token_vocab_id)

                    # add given node to the graph
                    # add new node
                    graph.add_nodes(1)
                    graph.ndata['h'][-1] = child_h
                    # add edge from current parent to the current child node
                    graph.add_edges([current_node_id], [nodeid_counter])
                    # if we want to add the reverse edge
                    graph.add_edges([nodeid_counter], [current_node_id])
                    # perform message passing after adding one child
                    # and update the ndata hidden tensor
                    graph.ndata['h'] = self.message_passing_network(graph, graph.ndata['h'])
            # print(f"decoded_list:{decoded_list}")
            out_graphs.append(graph)
        return out_graphs
