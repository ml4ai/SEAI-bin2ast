# sample test file to make sure the tree2tree model works as expected

import torch.optim

from utils.tree2tree import get_input_tree, get_target_tree
from models.encoders.tree_encoder import TreeEncoder
from models.decoders.tree_decoder import TreeDecoder
from models.tree2tree.tree2tree import TreeEncoderTreeDecoder

input_vocab = {"root": 0, "=": 1, "a": 2, "b": 3, "c": 4, "d": 5, "eos": 6}
output_vocab = {"root": 0, "eq": 1, "a": 2, "b": 3, "c": 4, "d": 5, "eos": 6}

h_dim = 256
emb_dim = 256
parent_feeding = False

input_tree = get_input_tree(input_vocab)
target_tree = get_target_tree(output_vocab)

print("Training...")


def main():
    epochs = 100
    encoder = TreeEncoder(in_dim=len(input_vocab), h_dim=h_dim, embedding_size=emb_dim)
    decoder = TreeDecoder(h_dim=h_dim, output_vocab=output_vocab, parent_feeding=parent_feeding)
    model = TreeEncoderTreeDecoder(encoder, decoder)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
    model.train()
    out_trees = None
    for epoch in range(epochs):  # loop over the dataset multiple times
        # list of graphs as input
        optimizer.zero_grad()
        loss = 0.0
        out_trees, loss = model([input_tree], [target_tree], loss)
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch + 1}, loss: {loss.item()}")

    print(f"ground truth number of nodes: {input_tree.total_number_of_nodes()}")
    out_tree = out_trees[0]
    print(f"predicted number of nodes before pruning: {out_tree.total_number_of_nodes()}")
    # prune / chunk tree after <eos> token
    out_tree.prune_nodes_after_eos()
    print(f"predicted number of nodes after pruning: {out_tree.total_number_of_nodes()}")
    # print output graph vocab_id: preorder traversal
    print(f"ground truth vocab ids: ", end=" ")
    print(f"{target_tree.root.vocab_id}", end=" ")
    print(f"{target_tree.root.left_child.vocab_id}", end=" ")
    print(f"{target_tree.root.right_child.vocab_id}", end=" ")
    print(f"{target_tree.root.left_child.left_child.vocab_id}", end=" ")
    print(f"{target_tree.root.left_child.right_child.vocab_id}", end=" ")
    print(f"{target_tree.root.right_child.left_child.vocab_id}", end=" ")
    print(f"{target_tree.root.right_child.right_child.vocab_id}", end=" ")

    print(f"\npredicted vocab_ids: ", end=" ")
    try:
        print(f"{out_tree.root.vocab_id}", end=" ")
        print(f"{out_tree.root.left_child.vocab_id}", end=" ")
        print(f"{out_tree.root.right_child.vocab_id}", end=" ")
        print(f"{out_tree.root.left_child.left_child.vocab_id}", end=" ")
        print(f"{out_tree.root.left_child.right_child.vocab_id}", end=" ")
        print(f"{out_tree.root.right_child.left_child.vocab_id}", end=" ")
        print(f"{out_tree.root.right_child.right_child.vocab_id}", end="\n")
    except AttributeError:
        pass


main()
