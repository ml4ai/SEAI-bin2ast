# util functions for testing tree2tree model
from utils.tree import BinaryTree


def get_input_tree(input_vocab):
    # input graph: test graph for testing TreeEncoder
    #                   root[node_id=0]
    #                     /           \
    #                    /             \
    #                   /               \
    #             =[node_id=1]        =[node_id=2]
    #               / \                  /      \
    #              /   \                /        \
    #             /     \              /          \
    #   a[node_id=3]  b[node_id=4]   c[node_id=5]  d[node_id=6]

    # add nodes in bfs: similar to what's done in code generation
    # because node ids are incremented that way and we search using node_ids
    nodeid_counter = 0
    input_tree = BinaryTree(vocab=input_vocab, root_id=nodeid_counter, root_val='root')  # node 0
    nodeid_counter += 1
    input_tree.add_left_child(parent_node_id=0, child_node_id=nodeid_counter, child_val="=")  # node 1
    nodeid_counter += 1
    input_tree.add_right_child(parent_node_id=0, child_node_id=nodeid_counter, child_val="=")  # node 2
    nodeid_counter += 1
    input_tree.add_left_child(parent_node_id=1, child_node_id=nodeid_counter, child_val="a")  # node 3
    nodeid_counter += 1
    input_tree.add_right_child(parent_node_id=1, child_node_id=nodeid_counter, child_val="b")  # node 4
    nodeid_counter += 1
    input_tree.add_left_child(parent_node_id=2, child_node_id=nodeid_counter, child_val="c")  # node 5
    nodeid_counter += 1
    input_tree.add_right_child(parent_node_id=2, child_node_id=nodeid_counter, child_val="d")  # node 6
    return input_tree


def get_target_tree(output_vocab):
    # output graph: test graph for testing TreeDecoder
    #                   root[node_id=0]
    #                     /           \
    #                    /             \
    #                   /               \
    #             eq[node_id=1]        eq[node_id=2]
    #               / \                  /      \
    #              /   \                /        \
    #             /     \              /          \
    #   a[node_id=3]  b[node_id=4]   c[node_id=5]  d[node_id=6]

    # add nodes in bfs: similar to what's done in code generation
    # because node ids are incremented that way and we search using node_ids
    nodeid_counter = 0
    target_tree = BinaryTree(vocab=output_vocab, root_id=nodeid_counter, root_val='root')  # node 0
    nodeid_counter += 1
    target_tree.add_left_child(parent_node_id=0, child_node_id=nodeid_counter, child_val="eq")  # node 1
    nodeid_counter += 1
    target_tree.add_right_child(parent_node_id=0, child_node_id=nodeid_counter, child_val="eq")  # node 2
    nodeid_counter += 1
    target_tree.add_left_child(parent_node_id=1, child_node_id=nodeid_counter, child_val="a")  # node 3
    nodeid_counter += 1
    target_tree.add_right_child(parent_node_id=1, child_node_id=nodeid_counter, child_val="b")  # node 4
    nodeid_counter += 1
    target_tree.add_left_child(parent_node_id=2, child_node_id=nodeid_counter, child_val="c")  # node 5
    nodeid_counter += 1
    target_tree.add_right_child(parent_node_id=2, child_node_id=nodeid_counter, child_val="d")  # node 6
    return target_tree
