# convert target sequences to tree and back to sequence and verify that everything
# is working correctly
import argparse
import yaml
from pickle_loader import load_pickled_data
from data_loader import get_loaders
from utils.tree import sequence_to_nary, nary_to_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="path to a config file")
args = parser.parse_args()

config_path = args.config
with open(config_path, 'r') as read_file:
    config = yaml.safe_load(read_file)

pickled_data_path = config["pickled_data_path"]
data_augmentation = config["data_augmentation"]

pickled_data = load_pickled_data(pickled_data_path=pickled_data_path)
train_input_data = pickled_data['train_input_data']
train_target_data = pickled_data['train_target_data']
val_input_data = pickled_data['val_input_data']
val_target_data = pickled_data['val_target_data']
test_input_data = pickled_data['test_input_data']
test_target_data = pickled_data['test_target_data']
input_lang = pickled_data["input_lang"]
output_lang = pickled_data['output_lang']

# add eoc for tree / graph generative models
output_lang.add_eoc()

train_loader, val_loader, test_loader = get_loaders(train_input_data, train_target_data,
                                                    val_input_data, val_target_data,
                                                    test_input_data, test_target_data,
                                                    input_lang=input_lang,
                                                    data_augmentation=data_augmentation,
                                                    batch_size=1)

print("Checking the output sequence to tree [and vice versa] conversion...")

max_nodes_nary = 0
max_nodes_binary = 0

for batch in train_loader:
    source = batch[0]
    target_list = batch[1].T[0].tolist()[1:-1]
    # target sequence
    target_str = [output_lang.index2token[key] for key in target_list]
    # convert target sequence to target nary tree
    target_tree_nary = sequence_to_nary(sequence=target_str,
                                        vocab=output_lang.token2index,
                                        eos_token=output_lang.eos_token)
    # convert target nary tree to target binary tree
    target_tree_binary = target_tree_nary.get_binary_tree_bfs()

    # find maximum number of nodes
    num_nodes_nary = target_tree_nary.total_number_of_nodes()
    if num_nodes_nary > max_nodes_nary:
        max_nodes_nary = num_nodes_nary

    num_nodes_binary = target_tree_binary.total_number_of_nodes()
    if num_nodes_binary > max_nodes_binary:
        max_nodes_binary = num_nodes_binary

    # convert binary tree back to nary tree
    target_tree_nary_reverse = target_tree_binary.get_nary_tree()
    # convert the reversed nary tree to a sequence
    target_str_reverse = nary_to_sequence(target_tree_nary_reverse)
    try:
        assert target_str == target_str_reverse
    except AssertionError:
        breakpoint()

print("All conversions are corect for training set...")
for batch in val_loader:
    source = batch[0]
    target_list = batch[1].T[0].tolist()[1:-1]
    # target sequence
    target_str = [output_lang.index2token[key] for key in target_list]
    # convert target sequence to target nary tree
    target_tree_nary = sequence_to_nary(sequence=target_str,
                                        vocab=output_lang.token2index,
                                        eos_token=output_lang.eos_token)
    # convert target nary tree to target binary tree
    target_tree_binary = target_tree_nary.get_binary_tree_bfs()

    # find maximum number of nodes
    num_nodes_nary = target_tree_nary.total_number_of_nodes()
    if num_nodes_nary > max_nodes_nary:
        max_nodes_nary = num_nodes_nary

    num_nodes_binary = target_tree_binary.total_number_of_nodes()
    if num_nodes_binary > max_nodes_binary:
        max_nodes_binary = num_nodes_binary

    # convert binary tree back to nary tree
    target_tree_nary_reverse = target_tree_binary.get_nary_tree()
    # convert the reversed nary tree to a sequence
    target_str_reverse = nary_to_sequence(target_tree_nary_reverse)
    try:
        assert target_str == target_str_reverse
    except AssertionError:
        breakpoint()

print("All conversions are correct for validation set...")
for batch in test_loader:
    source = batch[0]
    target_list = batch[1].T[0].tolist()[1:-1]
    # target sequence
    target_str = [output_lang.index2token[key] for key in target_list]
    # convert target sequence to target nary tree
    target_tree_nary = sequence_to_nary(sequence=target_str,
                                        vocab=output_lang.token2index,
                                        eos_token=output_lang.eos_token)
    # convert target nary tree to target binary tree
    target_tree_binary = target_tree_nary.get_binary_tree_bfs()

    # find maximum number of nodes
    num_nodes_nary = target_tree_nary.total_number_of_nodes()
    if num_nodes_nary > max_nodes_nary:
        max_nodes_nary = num_nodes_nary

    num_nodes_binary = target_tree_binary.total_number_of_nodes()
    if num_nodes_binary > max_nodes_binary:
        max_nodes_binary = num_nodes_binary

    # convert binary tree back to nary tree
    target_tree_nary_reverse = target_tree_binary.get_nary_tree()
    # convert the reversed nary tree to a sequence
    target_str_reverse = nary_to_sequence(target_tree_nary_reverse)
    try:
        assert target_str == target_str_reverse
    except AssertionError:
        breakpoint()

print(f"All conversions are corect for testing set...")
print(f"max_nodes_binary: {max_nodes_binary}")
print(f"max_nodes_nary: {max_nodes_nary}")
