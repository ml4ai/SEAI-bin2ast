"""
converts raw text files of input and output token sequences into tensors and stores
them in a dictionary as a pickle file so that we don't need to do the same costly
operation while training multiple models
"""

import yaml
import argparse
import os
from utils.lang import InputLanguage, OutputLanguage
import pickle
from utils.util import get_input_tokens_list_v3, get_output_tokens_list_v3
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to a config file")
args = parser.parse_args()

config_path = args.config_path
with open(config_path, 'r') as read_file:
    config = yaml.safe_load(read_file)

data_path = config["raw_data_path"]
save_path = config["pickled_data_path"]
create_corpora_combined = config["create_corpora_combined"]

corpora_combined_path = os.path.join(data_path + os.sep + "corpora_combined")
if create_corpora_combined:
    if os.path.isdir(corpora_combined_path):
        shutil.rmtree(corpora_combined_path)
    existing_folders = os.listdir(data_path)
    # remove .DS_Store
    for folder in existing_folders:
        if folder.startswith('.'):
            existing_folders.remove(folder)
    # create new folders
    os.mkdir(corpora_combined_path)
    input_directory = os.path.join(corpora_combined_path + os.sep + "tokens_input")
    os.mkdir(input_directory)
    output_directory = os.path.join(corpora_combined_path + os.sep + "tokens_output")
    os.mkdir(output_directory)
    # loop over folders and copy files
    for folder_name in existing_folders:
        input_path = data_path + os.sep + folder_name + os.sep + "tokens_input"
        input_files = os.listdir(input_path)
        for file in input_files:
            if file.endswith("__tokens.txt"):
                shutil.copy(input_path + os.sep + file, input_directory)
        output_path = data_path + os.sep + folder_name + os.sep + "tokens_output"
        output_files = os.listdir(output_path)
        for file in output_files:
            if file.endswith("--CAST.tcast"):
                shutil.copy(output_path + os.sep + file, output_directory)
    print("corpora_combined created!")

input_dir = corpora_combined_path + os.sep + "tokens_input"
output_dir = corpora_combined_path + os.sep + "tokens_output"
input_files = sorted(os.listdir(input_dir))

# take last 10k samples into test set
test_input_files = input_files[-10000:]
# take remaining 10k from last into validation set
val_input_files = input_files[-20000:-10000]
# take remaining data into training set
train_input_files = input_files[:-20000]

input_lang = config["input_lang"]
output_lang = config["output_lang"]
# create input and output lang
input_lang = InputLanguage(input_lang)
output_lang = OutputLanguage(output_lang)

# simple dict to store some stats
stats = {}
max_length = 0

# load training files and create training data and vocabulary
print("preparing training data...")
train_input_data = []
train_target_data = []
for file in train_input_files:
    target_file = file.strip().split("__")[0] + "--CAST.tcast"
    input_path = input_dir + "/" + file
    output_path = output_dir + "/" + target_file

    input_tokens_dict = get_input_tokens_list_v3(input_path)  # dict func_name : token_seq
    output_tokens_dict = get_output_tokens_list_v3(output_path)  # dict func_name : token_seq

    # need to fix this: this assert should be true:
    # assert len(input_tokens_dict) == len(output_tokens_dict)
    # assert input_tokens_dict.keys() == output_tokens_dict.keys()

    for func_name, token_seq_input in input_tokens_dict.items():
        input_tensor = input_lang.add_tokens(token_seq_input)
        try:
            token_seq_output = output_tokens_dict[func_name]
        except KeyError:
            raise Exception(f"Unable to find corresponding function name {func_name}\n\
                            of source file: {file}\n\
                            in target file: {target_file}")

        target_tensor = output_lang.add_tokens(token_seq_output)
        target_length = target_tensor.shape[0]
        if target_length > max_length:
            max_length = target_length
        train_input_data.append(input_tensor)
        train_target_data.append(target_tensor)

stats["max_length"] = max_length
# save the training data
print("saving train data...")
train_input_path = save_path + "/" + "train_input_data.pickle"
with open(train_input_path, 'wb') as write_file:
    pickle.dump(train_input_data, write_file)

train_target_path = save_path + "/" + "train_target_data.pickle"
with open(train_target_path, 'wb') as write_file:
    pickle.dump(train_target_data, write_file)

# save the input_lang and output_lang objects
input_lang_path = save_path + "/" + "input_lang.pickle"
output_lang_path = save_path + "/" + "output_lang.pickle"

print("saving input lang...")
with open(input_lang_path, 'wb') as write_file:
    pickle.dump(input_lang, write_file)

print("saving output lang...")
with open(output_lang_path, 'wb') as write_file:
    pickle.dump(output_lang, write_file)

# save the simple stats
stats["input_lang_tokens"] = input_lang.n_tokens
stats["output_lang_tokens"] = output_lang.n_tokens
stats_path = save_path + "/" + "stats.pickle"
with open(stats_path, 'wb') as write_file:
    pickle.dump(stats, write_file)

# save the input_val_data and target_val_data for validation dataset
print("preparing val data...")
# unknown tokens
val_input_unk = 0
val_target_unk = 0

val_input_data = []
val_target_data = []
for file in val_input_files:
    target_file = file.strip().split("__")[0] + "--CAST.tcast"
    input_path = input_dir + "/" + file
    output_path = output_dir + "/" + target_file

    input_tokens_dict = get_input_tokens_list_v3(input_path)  # dict func_name : token_seq
    output_tokens_dict = get_output_tokens_list_v3(output_path)  # dict func_name : token_seq

    # need to fix this: this assert should be true:
    # assert len(input_tokens_dict) == len(output_tokens_dict)
    # assert input_tokens_dict.keys() == output_tokens_dict.keys()

    for func_name, token_seq_input in input_tokens_dict.items():
        input_tensor, unk_count = input_lang.tensor_from_sequence(token_seq_input)
        val_input_unk += unk_count
        try:
            token_seq_output = output_tokens_dict[func_name]
        except KeyError:
            raise Exception(f"Unable to find corresponding function name {func_name}\n\
                            of source file: {file}\n\
                            in target file: {target_file}")

        target_tensor, unk_count = output_lang.tensor_from_sequence(token_seq_output)
        val_target_unk += unk_count
        val_input_data.append(input_tensor)
        val_target_data.append(target_tensor)

print(f"unk tokens in val input data: {val_input_unk}")
print(f"unk tokens in val target data: {val_target_unk}")

# save the training data
print("saving val data...")
val_input_path = save_path + "/" + "val_input_data.pickle"
with open(val_input_path, 'wb') as write_file:
    pickle.dump(val_input_data, write_file)

val_target_path = save_path + "/" + "val_target_data.pickle"
with open(val_target_path, 'wb') as write_file:
    pickle.dump(val_target_data, write_file)

# save the input_test_data and target_test_data for test dataset
print("preparing test data...")
# unk tokens in the test data
test_input_unk = 0
test_target_unk = 0

test_input_data = []
test_target_data = []
for file in test_input_files:
    target_file = file.strip().split("__")[0] + "--CAST.tcast"
    input_path = input_dir + "/" + file
    output_path = output_dir + "/" + target_file

    input_tokens_dict = get_input_tokens_list_v3(input_path)  # dict func_name : token_seq
    output_tokens_dict = get_output_tokens_list_v3(output_path)  # dict func_name : token_seq

    # need to fix this: this assert should be true:
    # assert len(input_tokens_dict) == len(output_tokens_dict)
    # assert input_tokens_dict.keys() == output_tokens_dict.keys()

    for func_name, token_seq_input in input_tokens_dict.items():
        input_tensor, unk_count = input_lang.tensor_from_sequence(token_seq_input)
        test_input_unk += unk_count
        try:
            token_seq_output = output_tokens_dict[func_name]
        except KeyError:
            raise Exception(f"Unable to find corresponding function name {func_name}\n\
                            of source file: {file}\n\
                            in target file: {target_file}")

        target_tensor, unk_count = output_lang.tensor_from_sequence(token_seq_output)
        test_target_unk += unk_count
        test_input_data.append(input_tensor)
        test_target_data.append(target_tensor)

print(f"unk tokens in test input data: {test_input_unk}")
print(f"unk tokens in test target data: {test_target_unk}")

# save the training data
print("saving test data...")
test_input_path = save_path + "/" + "test_input_data.pickle"
with open(test_input_path, 'wb') as write_file:
    pickle.dump(test_input_data, write_file)

test_target_path = save_path + "/" + "test_target_data.pickle"
with open(test_target_path, 'wb') as write_file:
    pickle.dump(test_target_data, write_file)

print(f"total training samples: {len(train_input_data)}")
print(f"total validation samples: {len(val_input_data)}")
print(f"total test samples: {len(test_input_data)}")
print("Done!")
