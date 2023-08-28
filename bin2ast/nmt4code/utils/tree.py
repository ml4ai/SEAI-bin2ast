# class to handle the conversion of sequence to a tree
# conversion of nary tree to binary and binary to nary
# note: the nary tree and binary tree uses unique node_id
# to identify each node and create a dictionary of node_id to the node
# messing up with this node_id may result in incorrect results

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        self.stack.pop()

    def is_empty(self):
        return len(self.stack) == 0

    def len(self):
        return len(self.stack)

    def second_last(self):
        return self.stack[-2]

    def last(self):
        return self.stack[-1]


class BinaryNode:
    def __init__(self, node_id, vocab_id=None, val=None, parent_node=None):
        # properties for general information and binary to nary conversion
        self.node_id = node_id
        self.val = val
        self.parent_node = parent_node
        self.left_child = None
        self.right_child = None
        # properties for tree encoder and decoder
        self.vocab_id = vocab_id
        # hidden state and context vector for each node
        self.h = None
        self.c = None
        self.e_t = None

    def is_root(self):
        return self.val == "main"


class BinaryTree:
    def __init__(self, vocab, root_id=0, root_val="main", eos_token="<eos>"):
        """
        vocab: vocabulary dictionary to get the vocab_id from the token name
        """
        self.vocab = vocab
        try:
            root_vocab_id = self.vocab[root_val]
        except KeyError:
            root_vocab_id = 0
        self.eos_token = eos_token
        self.root = BinaryNode(node_id=root_id, vocab_id=root_vocab_id, val=root_val)
        self.id_to_node = {root_id: self.root}

    def update_root(self, root_val):
        # update root_val and root_vocab_id
        vocab_id = self.vocab[root_val]
        self.root.vocab_id = vocab_id
        self.root.val = root_val

    def total_number_of_nodes(self):
        return len(self.id_to_node)

    def get_node_dict(self):
        """
        :return: dict of node id to node of a current tree
        """
        return self.id_to_node

    def prune_nodes_after_eos(self):
        """
        if we find a child with vocab_id of eos: remove that subtree
        :return: None
        """
        temp = {}
        for node_id, node in self.id_to_node.items():
            if node.vocab_id == self.vocab[self.eos_token]:
                break
            else:
                temp[node_id] = node
        self.id_to_node = temp

    def add_left_child(self, parent_node_id, child_node_id, child_val):
        parent_node = self.id_to_node[parent_node_id]

        try:
            vocab_id = self.vocab[child_val]
        except KeyError:
            vocab_id = None

        left_child = BinaryNode(node_id=child_node_id, vocab_id=vocab_id,
                                val=child_val, parent_node=parent_node)

        parent_node.left_child = left_child
        self.id_to_node[child_node_id] = left_child
        return left_child

    def add_right_child(self, parent_node_id, child_node_id, child_val):
        parent_node = self.id_to_node[parent_node_id]

        try:
            vocab_id = self.vocab[child_val]
        except KeyError:
            vocab_id = None

        right_child = BinaryNode(node_id=child_node_id, vocab_id=vocab_id,
                                 val=child_val, parent_node=parent_node)

        parent_node.right_child = right_child
        self.id_to_node[child_node_id] = right_child
        return right_child

    def get_label(self, node_id):
        """
        return vocab_id (label) of a given node number
        :param node_id: node id of the node to get the label for
        :return: either label for the given node or label for the eos token
        """
        try:
            label = self.id_to_node[node_id].vocab_id
        except KeyError:
            label = self.vocab[self.eos_token]
        return label

    def level_order(self):
        """
        print level order for binary tree
        :return: None
        """
        node_list = [self.root]
        node_val_list = list()
        while node_list:
            current_node = node_list.pop(0)
            node_val_list.append(current_node.val)
            if current_node.left_child is not None:
                node_list.append(current_node.left_child)
            if current_node.right_child is not None:
                node_list.append(current_node.right_child)
        return node_val_list

    def update_node_ids(self):
        """
        Give node ids to the binary tree based on level order [bfs order]
        also update the id to node dictionary
        """
        self.id_to_node = {}
        node_list = [self.root]
        nodeid_counter = 0
        while node_list:
            current_node = node_list.pop(0)
            current_node.node_id = nodeid_counter
            self.id_to_node.update({nodeid_counter: current_node})
            nodeid_counter += 1
            if current_node.left_child is not None:
                node_list.append(current_node.left_child)
            if current_node.right_child is not None:
                node_list.append(current_node.right_child)

    def add_eoc(self):
        """
        convert the None child to eoc child: we need this for generative model
        """
        node_list = [self.root]
        while node_list:
            current_node = node_list.pop(0)
            if current_node.left_child is not None:
                node_list.append(current_node.left_child)
            else:
                eoc_vocab_id = self.vocab["<eoc>"]
                current_node.left_child = BinaryNode(node_id=-1, vocab_id=eoc_vocab_id, val="<eoc>")

            if current_node.right_child is not None:
                node_list.append(current_node.right_child)
            else:
                eoc_vocab_id = self.vocab["<eoc>"]
                current_node.right_child = BinaryNode(node_id=-1, vocab_id=eoc_vocab_id, val="<eoc>")

        self.update_node_ids()

    def none_to_eoc(self):
        """
        some of the nodes during tree decoding are none
        convert them to <eoc> nodes
        """
        node_list = [self.root]
        while node_list:
            current_node = node_list.pop(0)
            if current_node.val is None:
                current_node.val = "<eoc>"
                current_node.vocab_id = self.vocab["<eoc>"]

            if current_node.left_child is not None:
                node_list.append(current_node.left_child)

            if current_node.right_child is not None:
                node_list.append(current_node.right_child)

    def get_nary_tree(self):
        """
        generate n-ary tree from current binary tree: reverse of nary - binary conversion
        left child: child[0]
        right child: next child of the parent
        :return: narytree
        """

        nry_tree = NaryTree(vocab=self.vocab,
                            root_id=0,
                            root_val=self.root.val,
                            eos_token=self.eos_token)

        nry_tree.add_child(parent_node_id=self.root.node_id,
                           child_node_id=self.root.left_child.node_id,
                           child_val=self.root.left_child.val)

        node_list = [self.root.left_child]
        # traverse the tree in bfs order
        while node_list:
            current_node = node_list.pop(0)
            if current_node.left_child is not None:
                left_child = current_node.left_child
                nry_tree.add_child(parent_node_id=current_node.node_id,
                                   child_node_id=left_child.node_id,
                                   child_val=left_child.val)
                node_list.append(left_child)
            if current_node.right_child is not None:
                right_child = current_node.right_child
                parent_node = current_node.parent_node
                # also update the parent node name for the right child
                right_child.parent_node = parent_node
                nry_tree.add_child(parent_node_id=parent_node.node_id,
                                   child_node_id=right_child.node_id,
                                   child_val=right_child.val)
                node_list.append(right_child)

        # note: the node id has been named from binary tree for easy nary tree construction
        # given them node_ids based on level order
        nry_tree.update_node_ids()
        return nry_tree


class NaryNode:
    def __init__(self, node_id, vocab_id=None, val=None):
        self.node_id = node_id
        self.vocab_id = vocab_id
        self.val = val
        self.children = []
        self.h = None
        self.c = None

    def add_child(self, child_node):
        self.children.append(child_node)


class NaryTree:
    def __init__(self, vocab, root_id=0, root_val="main", eos_token="<eos>"):
        """
        vocab: vocabulary dictionary to get the vocab_id from the token name
        """
        self.vocab = vocab

        try:
            root_vocab_id = self.vocab[root_val]
        except KeyError:
            root_vocab_id = 0

        self.eos_token = eos_token
        self.root = NaryNode(node_id=root_id, vocab_id=root_vocab_id, val=root_val)
        self.id_to_node = {root_id: self.root}

    def update_root(self, root_val):
        # update root vocab id and root val
        try:
            vocab_id = self.vocab[root_val]
        except KeyError:
            vocab_id = 0
        self.root.vocab_id = vocab_id
        self.root.val = root_val

    def total_number_of_nodes(self):
        return len(self.id_to_node)

    def add_child(self, parent_node_id, child_node_id, child_val, return_child=False):
        parent_node = self.id_to_node[parent_node_id]

        try:
            vocab_id = self.vocab[child_val]
        except KeyError:
            vocab_id = None

        child_node = NaryNode(node_id=child_node_id, vocab_id=vocab_id, val=child_val)
        parent_node.add_child(child_node)
        self.id_to_node[child_node_id] = child_node
        if return_child:
            return child_node

    def level_order(self):
        """
        start with root and print the nodes in level order
        :return: None
        """
        node_list = [self.root]
        node_val_list = []
        while node_list:
            current_node = node_list.pop(0)
            node_val_list.append(current_node.val)
            node_list.extend(current_node.children)
        return node_val_list

    def update_node_ids(self):
        """
        Give node ids to the nodes in level order
        also update the node id to node dictionary
        """
        self.id_to_node = {}
        node_list = [self.root]
        nodeid_counter = 0
        while node_list:
            current_node = node_list.pop(0)
            current_node.node_id = nodeid_counter
            self.id_to_node.update({nodeid_counter: current_node})
            nodeid_counter += 1
            node_list.extend(current_node.children)

    def child_list(self, parent_node_id):
        node = self.id_to_node[parent_node_id]
        return node.children

    def get_binary_tree_bfs(self):
        """
        create a new instance of binary tree from nary tree: right sibling becomes the right child
        :return: binary tree
        """

        bin_tree = BinaryTree(vocab=self.vocab,
                              root_id=0,
                              root_val=self.root.val,
                              eos_token=self.eos_token)

        right_sibling = {}
        child_list = self.child_list(self.root.node_id)
        prev_child = child_list[0]
        for child in child_list[1:]:
            right_sibling.update({prev_child: child})
            prev_child = child

        # add left child
        bin_tree.add_left_child(parent_node_id=self.root.node_id,
                                child_node_id=child_list[0].node_id,
                                child_val=child_list[0].val)
        while right_sibling:
            current_node = list(right_sibling.keys())[0]
            current_right_sibling = right_sibling[current_node]
            child_list = self.child_list(current_node.node_id)

            if len(child_list) > 1:
                prev_child = child_list[0]
                for child in child_list[1:]:
                    right_sibling.update({prev_child: child})
                    prev_child = child

            if len(child_list) > 0 and child_list[0].val != "<eoc>":
                bin_tree.add_left_child(parent_node_id=current_node.node_id,
                                        child_node_id=child_list[0].node_id,
                                        child_val=child_list[0].val)

            if current_right_sibling.val != "<eoc>":
                bin_tree.add_right_child(parent_node_id=current_node.node_id,
                                         child_node_id=current_right_sibling.node_id,
                                         child_val=current_right_sibling.val)

            right_sibling.pop(current_node)

        # add eoc child for NoneType nodes: we need this for generative model
        # this will also call update_nodes: and it will provide nodes with id that's in bfs order
        bin_tree.add_eoc()
        return bin_tree

    def add_eoc(self):
        """
        add eoc child: denotes the end of child for current parent
        we need this for generative models
        update node ids and id to name dict
        """
        node_list = [self.root]
        while node_list:
            current_node = node_list.pop(0)
            if current_node.val != "<eoc>":
                eoc_vocab_id = self.vocab["<eoc>"]
                eoc_child = NaryNode(node_id=-1, vocab_id=eoc_vocab_id, val="<eoc>")
                current_node.children.append(eoc_child)
                node_list.extend(current_node.children)
        # update the node ids and id to node dict
        self.update_node_ids()


def get_output_binary_tree_with_only_root(output_vocab, root_id, root_val, eos_token):
    out_tree = BinaryTree(vocab=output_vocab, root_id=root_id, root_val=root_val, eos_token=eos_token)
    return out_tree


def get_output_nary_tree_with_only_root(output_vocab, root_id, root_val, eos_token):
    out_tree = NaryTree(vocab=output_vocab, root_id=root_id, root_val=root_val, eos_token=eos_token)
    return out_tree


def create_nary_tree_1():
    # root has node_id: 0
    nodeid_counter = 0
    n_tree = NaryTree(vocab={}, root_id=nodeid_counter, root_val="a")  # node 0

    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val="b")  # node 1
    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val="c")  # node 2
    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val="d")  # node 3

    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=1, child_node_id=nodeid_counter, child_val="e")  # node 4

    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=2, child_node_id=nodeid_counter, child_val="f")  # node 5
    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=2, child_node_id=nodeid_counter, child_val="g")  # node 6
    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=2, child_node_id=nodeid_counter, child_val="h")  # node 7

    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=3, child_node_id=nodeid_counter, child_val="i")  # node 8
    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=3, child_node_id=nodeid_counter, child_val="j")  # node 9

    nodeid_counter = nodeid_counter + 1
    n_tree.add_child(parent_node_id=9, child_node_id=nodeid_counter, child_val="k")  # node 10

    n_tree.add_eoc()
    return n_tree


def create_nary_tree_2():
    nodeid_counter = 0
    n_tree = NaryTree(vocab={}, root_id=nodeid_counter, root_val='1')  # node 0

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val='2')  # node 1
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val='3')  # node 2
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val='4')  # node 3
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val='5')  # node 4

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=1, child_node_id=nodeid_counter, child_val='6')  # node 5
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=1, child_node_id=nodeid_counter, child_val='7')  # node 6

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=2, child_node_id=nodeid_counter, child_val='8')  # node 7

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=4, child_node_id=nodeid_counter, child_val='9')  # node 8
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=4, child_node_id=nodeid_counter, child_val='10')  # node 9
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=4, child_node_id=nodeid_counter, child_val='11')  # node 10

    n_tree.add_eoc()
    return n_tree


def create_nary_tree_3():
    nodeid_counter = 0
    n_tree = NaryTree(vocab={}, root_id=nodeid_counter, root_val='a')  # node 0

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val='b')  # node 1
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val='f')  # node 2
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=0, child_node_id=nodeid_counter, child_val='j')  # node 3

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=1, child_node_id=nodeid_counter, child_val='c')  # node 4
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=1, child_node_id=nodeid_counter, child_val='d')  # node 5
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=1, child_node_id=nodeid_counter, child_val='e')  # node 6

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=2, child_node_id=nodeid_counter, child_val='g')  # node 7
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=2, child_node_id=nodeid_counter, child_val='h')  # node 8

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=3, child_node_id=nodeid_counter, child_val='k')  # node 9
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=3, child_node_id=nodeid_counter, child_val='l')  # node 10
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=3, child_node_id=nodeid_counter, child_val='m')  # node 11
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=3, child_node_id=nodeid_counter, child_val='n')  # node 12

    nodeid_counter += 1
    n_tree.add_child(parent_node_id=11, child_node_id=nodeid_counter, child_val='p')  # node 13
    nodeid_counter += 1
    n_tree.add_child(parent_node_id=11, child_node_id=nodeid_counter, child_val='q')  # node 14

    n_tree.add_eoc()
    return n_tree


def sequence_to_nary(sequence, vocab, eos_token):
    """
    convert the given sequential version of AST into nary tree [graph] and return a tree
    sequence: list of tokens
    vocab: token2index dict
    """
    head_val_stack = Stack()
    head_id_stack = Stack()
    nodeid_counter = 0
    root_val = None
    next_head = False
    nry_tree = NaryTree(vocab=vocab, root_id=nodeid_counter, eos_token=eos_token)
    for token in sequence:
        if token == "(":
            next_head = True
        elif token == ")":
            head_val_stack.pop()
            head_id_stack.pop()
        elif next_head:
            if root_val is None:
                root_val = token
                nry_tree.update_root(root_val=root_val)
            head_val_stack.push(token)
            head_id_stack.push(nodeid_counter)
            if head_val_stack.len() > 1:
                child_val = head_val_stack.last()
                parent_node_id = head_id_stack.second_last()
                child_node_id = head_id_stack.last()
                nry_tree.add_child(parent_node_id=parent_node_id,
                                   child_node_id=child_node_id,
                                   child_val=child_val)
            nodeid_counter = nodeid_counter + 1
            next_head = False
        else:
            parent_node_id = head_id_stack.last()
            child_node_id = nodeid_counter
            child_val = token
            nry_tree.add_child(parent_node_id=parent_node_id,
                               child_node_id=child_node_id,
                               child_val=child_val)
            nodeid_counter = nodeid_counter + 1

    # add "eoc" child: denotes the end of child for current parent
    # we need this for generative model
    nry_tree.add_eoc()
    return nry_tree


def nary_to_sequence(nary):
    """
    convert the nary tree to sequence and return
    """
    r_sequence = ["("]

    def dfs(root, sequence):
        sequence.append(root.val)
        if root.children:
            for child in root.children:
                if "Var" in child.val or "Val" in child.val or 'print' in child.val:
                    sequence.append(child.val)
                elif "<eoc>" in child.val:
                    continue
                else:
                    sequence.append("(")
                    dfs(child, sequence)
                    sequence.append(")")

    dfs(nary.root, r_sequence)
    r_sequence.append(")")
    return r_sequence


if __name__ == '__main__':
    for test in range(3):
        test_name = "create_nary_tree_" + str(test + 1)
        test_function = locals()[test_name]
        nary_tree = test_function()
        val_list_original = nary_tree.level_order()
        binary_tree = nary_tree.get_binary_tree_bfs()
        nary_tree_reverse = binary_tree.get_nary_tree()
        val_list_reverse = nary_tree_reverse.level_order()
        assert val_list_original == val_list_reverse

    seq = ['(', 'main', '(', 'assign', 'Var0', 'Val0', ')', '(', 'assign', 'Var1', '(', 'Mult', '(',
           'BitOr', 'Var0', 'Val1', ')', 'Val2', ')', ')', '(', 'assign', 'Var2', 'Val3', ')', '(',
           'assign', 'Var3', 'Val4', ')', '(', 'assign', 'Var4', 'Val5', ')', '(', 'call', 'printf',
           'Val6', 'Var4', ')', '(', 'return', 'Val5', ')', ')']

    tree = sequence_to_nary(sequence=seq, vocab={}, eos_token="<eos>")
    seq_back = nary_to_sequence(tree)
    assert seq == seq_back
