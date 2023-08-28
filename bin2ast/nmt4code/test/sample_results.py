"""
sample some results from trained model on validation and test dataset
based on parameters from config file
store the results in a test file
sample config file: configs/sample_results.yaml
"""

import yaml
import argparse
import torch
from data.pickle_loader import load_pickled_data
from data.data_loader import get_loaders
from models.encoders.lstm_opt_encoder import Encoder
from models.decoders.lstm_opt_decoder import Decoder
from models.seq2seq.seq2seq import SequenceToSequence
import warnings
from utils.bleu import sample_result

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to a config file")
args = parser.parse_args()

config_path = args.config_path
with open(config_path, 'r') as read_file:
    config = yaml.safe_load(read_file)

gpu_index = config["gpu_index"]
device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
pickled_data_path = config["pickled_data_path"]
hidden_dim = config["hidden_dim"]
model_path = config["model_path"]
batch_size = config["batch_size"]
input_embedding_dim = config["input_embedding_dim"]
output_embedding_dim = config["output_embedding_dim"]
n_layers = config["n_layers"]
dropout = config["dropout"]
n = config["n"]
destination_path = config["destination_path"]

pickled_data = load_pickled_data(pickled_data_path=pickled_data_path)
stats = pickled_data['stats']
train_input_data = pickled_data['train_input_data']
train_target_data = pickled_data['train_target_data']
val_input_data = pickled_data['val_input_data']
val_target_data = pickled_data['val_target_data']
test_input_data = pickled_data['test_input_data']
test_target_data = pickled_data['test_target_data']
output_lang = pickled_data['output_lang']

max_length = stats["max_length"]
input_lang_unique_tokens = stats["input_lang_tokens"]
output_lang_unique_tokens = stats["output_lang_tokens"]

_, val_loader, test_loader = get_loaders(train_input_data, train_target_data,
                                         val_input_data, val_target_data,
                                         test_input_data, test_target_data,
                                         batch_size=batch_size)

enc = Encoder(input_dim=input_lang_unique_tokens, emb_dim=input_embedding_dim,
              hid_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

dec = Decoder(output_dim=output_lang_unique_tokens, emb_dim=output_embedding_dim,
              hid_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

model = SequenceToSequence(encoder=enc, decoder=dec, device=device).to(device)

model.load_state_dict(torch.load(model_path))

print(f"Evaluating...")
val_results = sample_result(model, val_loader, output_lang, device, n=n)
test_results = sample_result(model, test_loader, output_lang, device, n=n)

print(f"writing result to {destination_path}/sample.txt file")
# save results into sample.txt file
with open(destination_path + '/' + 'sample.txt', 'w') as write_file:
    write_file.write("Val samples...\n")
    for val_result in val_results:
        write_file.write(f"target\n: {val_result[0]}\n")
        write_file.write(f"model output\n: {val_result[1]}\n")

    write_file.write("\nTest samples...\n")
    for test_result in test_results:
        write_file.write(f"target\n: {test_result[0]}\n")
        write_file.write(f"model output\n: {test_result[1]}\n")

print("Done!")
