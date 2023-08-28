from typing import List, Dict
from dataclasses import dataclass, field
import os
import difflib
import networkx
from analyze_results import load_v2_results_data
import argparse
import yaml


@dataclass
class TCAST:
    """
    data class that holds
    tokens: list of tokens in the tcast file
    vars: dictionary of variables (var#): actual variable name
    vals: dictionary of values (val#): actual value
    """
    tokens: List[str] = None
    vars: Dict[str, str] = field(default_factory=dict)
    vals: Dict[str, str] = field(default_factory=dict)


def tcast_to_tree(tcast: TCAST):
    """
    generate tree from tcast: sequence created with DFS (preorder traversal)
    :param tcast: TCAST instance
    :return: networkx digrah
    """
    G = networkx.DiGraph()
    node_counter = 0
    node_stack = list()
    next_head = False
    for elm in tcast.tokens:
        if elm == '(':
            next_head = True
        elif elm == ')':
            if len(node_stack) > 1:
                # edge between parent head to child head
                G.add_edge(node_stack[-2], node_stack[-1])
            node_stack.pop()
        elif next_head:
            node_stack.append(f'{node_counter}@{elm}')
            node_counter += 1
            next_head = False
        elif elm.startswith('Var'):
            if elm in tcast.vars:
                var_str = f'{node_counter}@{elm}={tcast.vars[elm]}'
            else:
                var_str = f'{node_counter}@{elm}'
            node_counter += 1
            # edge between parent head to var#
            G.add_edge(node_stack[-1], var_str)
        elif elm.startswith('Val'):
            if elm in tcast.vals:
                val_str = f'{node_counter}@{elm}={tcast.vals[elm]}'
            else:
                val_str = f'{node_counter}@{elm}'
            node_counter += 1
            # edge between parent head to val#
            G.add_edge(node_stack[-1], val_str)
    return G


def parse_tcast_file(filepath: str) -> TCAST:
    """
    parst tcast file and generate TCAST data class
    :param filepath: path to tcast file
    :return: TCAST
    """
    tcast = TCAST()
    in_vars_p = False
    in_vals_p = False
    with open(filepath, 'r') as tcast_file:
        for line in tcast_file.readlines():
            if tcast.tokens is None:
                tcast.tokens = line.strip('\n ').split(' ')
            elif line.startswith('--------- VARIABLES ---------'):
                in_vars_p = True
            elif line.startswith('--------- VALUES ---------'):
                in_vars_p = False
                in_vals_p = True
            elif in_vars_p:
                tcast.vars[f'Var{len(tcast.vars)}'] = line.strip('\n ')
            elif in_vals_p:
                tcast.vals[f'Val{len(tcast.vals)}'] = line.strip(' \n').replace(':', '>').replace('\\n', '')
    return tcast


def save_graph_dot(g, filepath):
    """
    save networkx graph to given path
    :param g: networkx graph
    :param filepath: destination path
    :return: None
    """
    networkx.drawing.nx_pydot.write_dot(g, filepath)


def parse_and_save_tcast_dot(filepath):
    """
    parse the tcast file and conver it into networkx graph and then save the dot file
    :param filepath: path to .tcast file
    :return: None
    """
    tcast = parse_tcast_file(filepath)
    G = tcast_to_tree(tcast)
    dot_filename = filepath.split(os.sep)[-1].split('--')[0] + '--tCAST.dot'
    dot_filepath = filepath.rsplit(os.sep, 1)[0] + os.sep + dot_filename
    save_graph_dot(G, dot_filepath)


def difflib_example():
    cases = [(('a', 'b', 'c', 'e'), ('a', 'b', 'd', 'f'))]
    for a, b in cases:
        print(f'{a} => {b}')
        for i, s in enumerate(difflib.ndiff(a, b)):
            if s[0] == ' ':
                continue
            elif s[0] == '-':
                print(f'Delete {s[-1]} from position {i}: {s}')
            elif s[0] == '+':
                print(f'Add {s[-1]} to position {i}: {s}')
        print()


def diff_tcast(tcast_target, tcast_predicted, name, save_path, bleu_score):
    """
    generate diference between two target_sequence and model_output using difflib.ndiff
    :param tcast_target: target tcast
    :param tcast_predicted: predicted tcast
    :param name: name of test sample
    :param save_path: save directory
    :param bleu_score: bleu score between tcast_target and tcast_predicted
    :return: None
    """
    G = tcast_to_tree(TCAST(tokens=tcast_target[1:-1]))
    save_graph_dot(G, save_path + os.sep + name + '--tCAST.dot')

    # save to file instead of stdout
    file_path = save_path + os.sep + name + '--diff.txt'
    with open(file_path, 'w') as write_file:
        write_file.write('predicted:\n')
        write_file.write(str(tcast_predicted) + "\n")
        write_file.write('target:\n')
        write_file.write(str(tcast_target) + "\n\n")
        # difflib.ndiff() produces '?': we need to remove them so that the positions aligns better
        index = 0
        # total changes required to convert tcast_predicted to tcast_target
        # the difflib.ndiff doesn't provide minimal changes required
        total_changes_required = 0
        for _, s in enumerate(difflib.ndiff(tcast_predicted, tcast_target)):
            if s[0] == '?':
                # these are hints from difflib.ndiff and we can ignore them
                continue
            elif s[0] == ' ':
                index += 1
                continue
            elif s[0] == '-':
                write_file.write(f'Delete {s.split(" ")[-1]} from position {index}: {s}\n')
                index += 1
                total_changes_required += 1
            elif s[0] == '+':
                write_file.write(f'Add {s.split(" ")[-1]} to position {index}: {s}\n')
                index += 1
                total_changes_required += 1

        print(f"total changes required: {total_changes_required}")
        write_file.write(f"\ntotal changes required: {total_changes_required}\n")
        write_file.write(f"\nbleu_score: {bleu_score}")


def diff_tcast_for(data, name, save_path):
    """
    generate diff for given name sample
    :param data: dictionary of name: (bleu_score, target_sequence, model_output)
    :param name:name of the sample
    :param save_path: save path directory
    :return: None
    """
    score, target, predicted = data[name]
    print(f'BLEU score for {name}: {score}')
    diff_tcast(target, predicted, name, save_path, score)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, 'r') as read_file:
        config = yaml.safe_load(read_file)

    # difflib_example()
    # tcast_path = config['tcast_path']
    # parse_and_save_tcast_dot(tcast_path)
    seq2seq_txt_files_no_values = config['seq2seq_txt_files_no_values']
    save_path = config["save_path"]
    test_files = config['test_files']
    bleu_data = load_v2_results_data(seq2seq_txt_files_no_values)

    for test_file in test_files:
        diff_tcast_for(bleu_data, test_file, save_path)

    print('DONE')


if __name__ == '__main__':
    main()
