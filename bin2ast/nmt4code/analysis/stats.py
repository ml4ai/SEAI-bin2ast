import argparse
import random
import yaml
from analyze_results import load_v2_results_data
from seq2tree import TCAST, tcast_to_tree, save_graph_dot
import os
import networkx
import pathlib


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


def is_structured(model_output: list) -> bool:
    """
    check if the model output can be converted back to tree from sequence
    :param model_output: output from model
    :return: bool
    """
    model_output = model_output[1:-1]
    stack = Stack()
    for item in model_output:
        if item == '(':
            stack.push(item)
        elif item == ')':
            if stack.len() == 0:
                return False
            else:
                stack.pop()

    if stack.is_empty():  # possible candidate for structured
        # It passes  num of ( ==  num of ): easy to evaluate
        # Try to generate actual tree
        try:
            _ = tcast_to_tree(TCAST(tokens=model_output[1:-1]))  # takes some time to evaluate
        except IndexError:
            return False
        return True
    else:
        return False


def graph_difference(filename: str, target_sequence: list, model_output: list, save_path: str,
                     save_graph_distance: bool = False, highlight_nodes: bool = False):
    """
    plot graph difference between target graph and output graph
    :param filename: path to file name
    :param target_sequence: ground truth sequence
    :param model_output: output from model
    :param save_path: path to save graph difference
    :param save_graph_distance: save the graph distance to txt file or not
    :param highlight_nodes: bool to either highligh the node or not
    :return: None
    """
    G_target = tcast_to_tree(TCAST(tokens=target_sequence[1:-1]))
    G_output = tcast_to_tree(TCAST(tokens=model_output[1:-1]))
    save_path = save_path + os.sep + filename
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    if highlight_nodes:
        # TODO: tried with difflib.ndiff: it won't work
        # TODO: try other algorithms as well instead of difflib.ndiff
        # let's save original for now
        save_graph_dot(G_target, save_path + os.sep + filename + '--tCAST-target.dot')
        save_graph_dot(G_output, save_path + os.sep + filename + '--tCAST-predicted.dot')
    else:
        save_graph_dot(G_target, save_path + os.sep + filename + '--tCAST-target.dot')
        save_graph_dot(G_output, save_path + os.sep + filename + '--tCAST-predicted.dot')
    # plot the difference graph as well
    if save_graph_distance:
        txt_file = save_path + os.sep + filename + ".txt"
        with open(txt_file, 'w') as write_file:
            graph_distane = networkx.graph_edit_distance(G_target, G_output)
            write_file.write(f"graph_distance: {graph_distane}")


def generate_stats(txt_file_location, save_path, save_graph_distance, generate_all_structured, highlight_nodes):
    """
    generate stats about the txt files generated from trained model
    :param txt_file_location: location of txt files
    :param save_path: path to save the results to
    :param save_graph_distance: bool to determine if we want to save the graph distance
    :param generate_all_structured: bool to determine if we want to generate CAST for all possible sequences
    :param highlight_nodes: bool to determine if we want to highlight the node difference from the predicted
    CAST tree with respect to actual CAST tree
    :return: None
    """
    print(f'stats from :{txt_file_location}')
    bleu_data = load_v2_results_data(txt_file_location, return_exact_match=True)
    total = len(bleu_data)
    exact = []  # files that have exact match
    unstructured = []  # files that can not be converted into trees
    structured = []  # Files that can be converted to trees but are not exact match

    for file_name, values in bleu_data.items():
        bleu_score, target_sequence, model_output, exact_match = values
        if exact_match:
            exact.append(file_name)
            continue
        if is_structured(model_output):
            structured.append(file_name)
        else:
            unstructured.append(file_name)

    assert len(exact) + len(unstructured) + len(structured) == total
    print(f'exact match: {len(exact)} / {total}')
    print(f'unstructured: {len(unstructured)} / {total}')
    print(f'structured: {len(structured)} / {total}')
    print()
    print(f'exact match: {round((len(exact) / total) * 100.0, 2)}%')
    print(f'unstructured: {round((len(unstructured) / total) * 100.0, 2)}%')
    print(f'structured: {round((len(structured) / total) * 100.0, 2)}%')

    # from structured output: filter if the errors are simple like different variable names or others
    # save one sample for structured output
    if generate_all_structured:
        print('Generating trees...')
        for filename in structured:
            _, target_sequence, model_output, _ = bleu_data[filename]
            graph_difference(filename, target_sequence, model_output, save_path, save_graph_distance,
                             highlight_nodes)
    else:
        random_file = random.choice(structured)
        _, target_sequence, model_output, _ = bleu_data[random_file]
        graph_difference(random_file, target_sequence, model_output, save_path, save_graph_distance,
                         highlight_nodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, 'r') as read_file:
        config = yaml.safe_load(read_file)

    # no_values
    seq2seq_txt_files_no_values = config['seq2seq_txt_files_no_values']
    save_path_seq2seq_no_values = config["save_path_seq2seq_no_values"]
    attention_txt_files_no_values = config['attention_txt_files_no_values']
    save_path_attention_no_values = config['save_path_attention_no_values']

    # with_values
    seq2seq_txt_files_with_values = config['seq2seq_txt_files_with_values']
    save_path_seq2seq_with_values = config['save_path_seq2seq_with_values']
    attention_txt_files_with_values = config['attention_txt_files_with_values']
    save_path_attention_with_values = config['save_path_attention_with_values']

    # config parameters
    save_graph_distance = config['save_graph_distance']
    generate_all_structured = config['generate_all_structured']
    highlight_nodes = config['highlight_nodes']

    # no values
    generate_stats(seq2seq_txt_files_no_values, save_path_seq2seq_no_values, save_graph_distance,
                   generate_all_structured, highlight_nodes)
    generate_stats(attention_txt_files_no_values, save_path_attention_no_values, save_graph_distance,
                   generate_all_structured, highlight_nodes)
    # with values
    generate_stats(seq2seq_txt_files_with_values, save_path_seq2seq_with_values, save_graph_distance,
                   generate_all_structured, highlight_nodes)
    generate_stats(attention_txt_files_with_values, save_path_attention_with_values, save_graph_distance,
                   generate_all_structured, highlight_nodes)

    print('Done!')


if __name__ == '__main__':
    main()
