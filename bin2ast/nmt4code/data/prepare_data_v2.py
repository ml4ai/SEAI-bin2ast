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
from utils.util import get_input_tokens_list_v2, get_output_tokens_list_v2

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to a config file")
args = parser.parse_args()

config_path = args.config_path
with open(config_path, 'r') as read_file:
    config = yaml.safe_load(read_file)

data_path = config["raw_data_path"]
save_path = config["pickled_data_path"]

input_dir = data_path + "/corpora_combined/tokens_input"
output_dir = data_path + "/corpora_combined/tokens_output"
input_files = sorted(os.listdir(input_dir))

# take last 10k samples into test set
test_input_files = input_files[-10000:]
# take remaining 10k from last into validation set
val_input_files = input_files[-20000:-10000]
# take remaining data into training set
train_input_files = input_files[:-20000]

input_lang = config["input_lang"]
output_lang = config["output_lang"]

# get actual values
with_values = config["with_values"]

# need to fix the hex values from tokens file or not
fix_hex_values = config["fix_hex_values"]

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

    # use get_input_tokens_list_v2_old for with values experiment
    # v2 dataset doesn't have globals: it only has a list of token list
    # and a dictionary of _v#(values) ---> actual values
    input_tokens_list, input_val_dict = get_input_tokens_list_v2(input_path, fix_hex_values=fix_hex_values)
    # if with_values, substitute the placeholders to actual values with each digit in the separate
    # place: 123.0 --> "1" "2" "3" "." "0"
    if with_values:
        new_input_tokens_list = []
        for token in input_tokens_list:
            if token in input_val_dict:
                value = input_val_dict[token]
                if value == "Answer":
                    new_input_tokens_list.append(value)
                else:
                    for item in value:
                        new_input_tokens_list.append(item)
            else:
                new_input_tokens_list.append(token)
        input_tokens_list = new_input_tokens_list

    input_tensor = input_lang.add_tokens(input_tokens_list)

    # use get_output_tokens_list_v2_old for with value experiment
    output_tokens_list, output_val_dict = get_output_tokens_list_v2(output_path)
    # if with_values, substitute the placeholders to actual values with each digit in the separate
    # place: 123.0 --> "1" "2" "3" "." "0"
    if with_values:
        new_output_tokens_list = []
        for token in output_tokens_list:
            if token in output_val_dict:
                value = output_val_dict[token]
                if value == "Answer":
                    new_output_tokens_list.append(value)
                else:
                    for item in value:
                        new_output_tokens_list.append(item)
            else:
                new_output_tokens_list.append(token)
        output_tokens_list = new_output_tokens_list

    target_tensor = output_lang.add_tokens(output_tokens_list)

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

    input_tokens_list, input_val_dict = get_input_tokens_list_v2(input_path, fix_hex_values=fix_hex_values)
    # if with_values, substitute the placeholders to actual values with each digit in the separate
    # place: 123.0 --> "1" "2" "3" "." "0"
    if with_values:
        new_input_tokens_list = []
        for token in input_tokens_list:
            if token in input_val_dict:
                value = input_val_dict[token]
                if value == "Answer":
                    new_input_tokens_list.append(value)
                else:
                    for item in value:
                        new_input_tokens_list.append(item)
            else:
                new_input_tokens_list.append(token)
        input_tokens_list = new_input_tokens_list

    input_tensor, unk_count = input_lang.tensor_from_sequence(input_tokens_list)
    val_input_unk += unk_count

    output_tokens_list, output_val_dict = get_output_tokens_list_v2(output_path)
    # if with_values, substitute the placeholders to actual values with each digit in the separate
    # place: 123.0 --> "1" "2" "3" "." "0"
    if with_values:
        new_output_tokens_list = []
        for token in output_tokens_list:
            if token in output_val_dict:
                value = output_val_dict[token]
                if value == "Answer":
                    new_output_tokens_list.append(value)
                else:
                    for item in value:
                        new_output_tokens_list.append(item)
            else:
                new_output_tokens_list.append(token)
        output_tokens_list = new_output_tokens_list

    target_tensor, unk_count = output_lang.tensor_from_sequence(output_tokens_list)
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

    input_tokens_list, input_val_dict = get_input_tokens_list_v2(input_path, fix_hex_values=fix_hex_values)
    # if with_values, substitute the placeholders to actual values with each digit in the separate
    # place: 123.0 --> "1" "2" "3" "." "0"
    if with_values:
        new_input_tokens_list = []
        for token in input_tokens_list:
            if token in input_val_dict:
                value = input_val_dict[token]
                if value == "Answer":
                    new_input_tokens_list.append(value)
                else:
                    for item in value:
                        new_input_tokens_list.append(item)
            else:
                new_input_tokens_list.append(token)
        input_tokens_list = new_input_tokens_list

    input_tensor, unk_count = input_lang.tensor_from_sequence(input_tokens_list)
    test_input_unk += unk_count

    output_tokens_list, output_val_dict = get_output_tokens_list_v2(output_path)
    # if with_values, substitute the placeholders to actual values with each digit in the separate
    # place: 123.0 --> "1" "2" "3" "." "0"
    if with_values:
        new_output_tokens_list = []
        for token in output_tokens_list:
            if token in output_val_dict:
                value = output_val_dict[token]
                if value == "Answer":
                    new_output_tokens_list.append(value)
                else:
                    for item in value:
                        new_output_tokens_list.append(item)
            else:
                new_output_tokens_list.append(token)
        output_tokens_list = new_output_tokens_list

    target_tensor, unk_count = output_lang.tensor_from_sequence(output_tokens_list)
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

print("Done!")
