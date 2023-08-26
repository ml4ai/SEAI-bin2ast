"""
calculate the distribution of input tokens and output tokens
"""

import yaml
import argparse
import os
from collections import defaultdict
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to a config file")
args = parser.parse_args()

config_path = args.config_path
with open(config_path, 'r') as read_file:
    config = yaml.safe_load(read_file)

data_path = config["raw_data_path"]
destination_path = config["destination_path"]

input_dir = data_path + "/" + "tokens_input"
output_dir = data_path + "/" + "tokens_output"
input_files = sorted(os.listdir(input_dir))

input_tokencount = defaultdict(int)
output_tokencount = defaultdict(int)


print("counting...")
for file in input_files:
    target_file = file.strip().split("__")[0] + "--CAST.tcast"
    input_path = input_dir + "/" + file
    output_path = output_dir + "/" + target_file

    with open(input_path, 'r') as read_file:
        tokens = read_file.readline().strip()
        tokens = tokens.replace("'", "")
        tokens_list = tokens.split(",")
        for item in tokens_list:
            token = item.strip()
            input_tokencount[token] += 1

    with open(output_path, 'r') as read_file:
        tokens = read_file.readline().strip()
        tokens_list = tokens.split()
        for item in tokens_list:
            token = item.strip()
            output_tokencount[token] += 1

print(f"input_tokencount: {input_tokencount}")
print(f"output_tokencount: {output_tokencount}")

# save the results

with open(destination_path + '/' + 'input_count.pickle', 'wb') as write_file:
    pickle.dump(input_tokencount, write_file)

with open(destination_path + '/' + 'output_count.pickle', 'wb') as write_file:
    pickle.dump(output_tokencount, write_file)

print("Done!")
