import os
import ast
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pickle
import tqdm
import pathlib
import yaml
import argparse
import numpy as np
import multiprocessing


def load_tokens_input_all(filepath: str) -> Dict:
    """
    load all_input.pickle file
    :param filepath: path to input_all.pickle
    :return: Dict{filename: input ghidra tokens}
    """
    print('START load_tokens_input_all')
    with open(filepath, 'rb') as tia_file:
        tokens_input_all = pickle.load(tia_file)
    print(f'DONE load_tokens_input_all {len(tokens_input_all)}')
    return tokens_input_all


def parse_results_file(filepath: str) -> Tuple[str, float, List[str], List[str], int, bool]:
    """
    parse each txt file and return the contents of that file
    :param filepath: path to txt file for each test sample
    :return: list[filename, bleu_score, target_sequence, model_output, exact_match]
    """
    with open(filepath, 'r') as txt_file:
        for line in txt_file:
            if line.startswith('input file: '):
                name = line.rsplit('__')[0].split(':')[1].strip()
            elif line.startswith('bleu score: '):
                score = float(line.split(':')[1].strip())
            elif line.startswith('target sequence:'):
                target_sequence = next(txt_file).strip()
                target_sequence = ast.literal_eval(target_sequence)
            elif line.startswith('model output:'):
                model_output = next(txt_file).strip()
                model_output = ast.literal_eval(model_output)
            elif line.startswith('seq_length:'):
                seq_length = int(line.split(':')[1].strip())
            elif line.startswith('exact_match:'):
                match_string = line.split(':')[1].strip()
                exact_match = True if match_string == "true" else False
    return name, score, target_sequence[0], model_output, seq_length, exact_match


def worker(files_list, return_dict, return_exact_match=False):
    """
    worker class for multiprocessing: loads data from given chunk
    :param files_list: files in the given chunk [file paths]
    :param return_dict: common return dictionary
    :param return_exact_match: if we want to return (score, target_sequence, model_output, exact_match)
    :return: Dictionary of filename: (bleu_score, target_sequence, model_output)
    """
    for filepath in files_list:
        # filepath = path_v2_results + os.sep + f
        name, score, target_sequence, model_output, seq_length, exact_match \
            = parse_results_file(filepath)
        if return_exact_match:
            return_dict[name] = (score, target_sequence, model_output, exact_match)
        else:
            return_dict[name] = (score, target_sequence, model_output)


def load_results_data(path_results: str, return_exact_match=False) -> Dict:
    """
    parse generated txt files
    :param path_results: path to txt files
    :param return_exact_match: if we want to return (score, target_sequence, model_output, exact_match)
    :return: Dict{file_name: (score, target_sequence, model_output)
    """
    print(f'START: load_results_data : {path_results}')
    files = []
    folders = os.listdir(path_results)
    for folder in folders:
        txt_files = os.listdir(path_results + os.sep + folder)
        for txt_file in txt_files:
            if txt_file.endswith(".txt"):
                files.append(path_results + os.sep + folder + os.sep + txt_file)

    manager = multiprocessing.Manager()
    data = manager.dict()
    n = multiprocessing.cpu_count()

    # create n chunks from files
    file_chunks = [files[i::n] for i in range(n)]
    jobs = []
    for i in range(n):
        p = multiprocessing.Process(target=worker, args=(file_chunks[i], data, return_exact_match))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    print(f'DONE: load_results_data {len(data)}')
    return data


def filter_models_by_bleu_range(data: Dict, low, high) -> List[Tuple]:
    """
    filter model name with bleu score between low and high
    :param data: Dict{filename: (score, target_sequence, model_output)
    :param low: low value
    :param high: high value
    :return: List[(model_name, length of model output)]
    """
    model_names = list()
    for name, (score, _, model_output) in data.items():
        if low <= score <= high:
            model_names.append((name, len(model_output)))
    return model_names


def plot_scores_histogram(data: Dict, save_filepath=None) -> None:
    """
    plot the histogram of bleu score vs frequency
    :param data: Dict{filename: (score, target_sequence, model_output)
    :param save_filepath: histogram filepath
    :return: None
    """
    scores = [score for score, _, _ in data.values()]
    freq, _, _ = plt.hist(scores, bins=50, alpha=0.85)
    plt.xlabel('BLUE Scores')
    plt.ylabel('Frequency')
    plt.title(f'BLUE Score Distribution, total={len(scores)}')
    plt.grid(True)
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, np.max(freq) + 100))
    if save_filepath:
        plt.savefig(save_filepath, format='pdf')


def plot_scores_by_length(data_bleu: Dict,  # data_input_tokens: Dict,
                          save_filepath_target: str = None):
    # save_filepath_input: str = None,
    # save_filepath_input_by_target_len: str = None,
    # save_filepath_target_to_pred_len: str = None) -> None:
    """
    plot various statistics:
    [1] target sequence length (ground truth) vs bleu score
    [2] input sequence length (ground truth) vs bleu score
    [3] input sequence length (ground truth) vs target sequence length (ground truth)
    [4] target sequence length (ground truth) vs predicted sequence length (model output)
    :param data_bleu: Dict{filename: (score, target_sequence, model_output)
    # :param data_input_tokens: Dict{filename: input ghidra tokens}
    :param save_filepath_target: save path for [1]
    # :param save_filepath_input: save path for [2]
    # :param save_filepath_input_by_target_len: save path for [3]
    # :param save_filepath_target_to_pred_len: save path for [4]
    :return: None
    """
    scores = list()
    # input_lengths = list()
    target_lengths = list()
    predicted_lengths = list()

    print('START plot_scores_by_length - gathering data')
    for name, (score, target_sequence, model_output) in tqdm.tqdm(data_bleu.items()):
        scores.append(score)
        # input_lengths.append(len(data_input_tokens[name]))
        target_lengths.append(len(target_sequence))
        predicted_lengths.append(len(model_output))
    print('DONE plot_scores_by_length - gathering data')

    my_dpi = 96
    plt.figure(figsize=(1200 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    plt.scatter(target_lengths, scores, s=3.0, alpha=0.75)
    plt.xlabel('Target CAST token sequence length')
    plt.ylabel('BLEU Score')
    plt.title('Target Sequence Length by BLEU Score')
    if save_filepath_target:
        plt.savefig(save_filepath_target, format='pdf')

    # plt.figure(figsize=(1200 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    # plt.scatter(input_lengths, scores, s=3.0, alpha=0.75)
    # plt.xlabel('Input Ghidra IR token sequence length')
    # plt.ylabel('BLEU Score')
    # plt.title('Input Sequence Length by BLEU Score')
    # if save_filepath_input:
    #     plt.savefig(save_filepath_input, format='pdf')

    # plt.figure()
    # plt.scatter(input_lengths, target_lengths, s=3.0, alpha=0.75)
    # plt.xlabel('Input Ghidra IR token sequence length')
    # plt.ylabel('Target CAST token sequence length')
    # plt.title('Input to Target token lengths')
    # if save_filepath_input:
    #     plt.savefig(save_filepath_input_by_target_len, format='pdf')

    plt.figure()
    plt.scatter(target_lengths, predicted_lengths, s=3.0, alpha=0.75)
    plt.xlabel('Target CAST token sequence length')
    plt.ylabel('Predicted CAST token sequence length')
    plt.title('Target to Predicted token lengths')
    # if save_filepath_input:
    #    plt.savefig(save_filepath_target_to_pred_len, format='pdf')


def main(root_dir, dst_dir):  # data_input_tokens=None
    """
    main script to generate plots
    :param root_dir: source directory for txt files generated from trained model
    :param dst_dir: path to store the generated plots
    # :param data_input_tokens: Dict{file_name: input ghidra tokens}
    :return: None
    """

    pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)

    data_bleu = load_results_data(root_dir)

    plot_scores_histogram(data_bleu, os.path.join(dst_dir, 'bleu_score_dist.pdf'))

    plot_scores_by_length(
        data_bleu,  # data_input_tokens,
        save_filepath_target=os.path.join(dst_dir, 'bleu_scores_by_target_len_scatter.pdf'),
        # save_filepath_input=os.path.join(dst_dir, 'bleu_scores_by_input_len_scatter.pdf'),
        # save_filepath_input_by_target_len=os.path.join(dst_dir, 'input_by_target_len_scatter.pdf'),
        # save_filepath_target_to_pred_len=os.path.join(dst_dir, 'target_to_predicted_len_scatter.pdf')
    )

    success = filter_models_by_bleu_range(data_bleu, low=1, high=1)
    print(f'succeeded: {len(success)}')
    failed = filter_models_by_bleu_range(data_bleu, low=0, high=0)
    print(f'failed: {len(failed)}')
    # part_0p9 = filter_models_by_bleu_range(data_bleu, low=0.9, high=0.91)
    # print(f'[0.9, 0.91]]: {len(part_0p9)} {part_0p9}')
    # part_0p8 = filter_models_by_bleu_range(data_bleu, low=0.8, high=0.81)
    # print(f'[0.8, 0.81]]: {len(part_0p8)} {part_0p8}')
    # part_0p6 = filter_models_by_bleu_range(data_bleu, low=0.6, high=0.61)
    # print(f'[0.6, 0.61]]: {len(part_0p6)} {part_0p6}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, 'r') as read_file:
        config = yaml.safe_load(read_file)

    # token_input_all_path = config["token_input_all"]
    # _data_input_tokens = load_tokens_input_all(token_input_all_path)

    folder_path_v3 = config["folder_path_v3"]
    destination_path = config["destination_path"]

    # generate report
    # no_values
    # seq2seq
    main(root_dir=folder_path_v3,
         dst_dir=destination_path)

    print("Done!")
