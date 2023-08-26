"""
generate one text file for each of the test sample with following information
target
output_from_model
bleu_score
output_from_model
bleu score
...
beam width times
also store: max bleu score: maximum bleu score among beam_width samples
sample config file: configs/test_bleu.yaml
"""

import yaml
import argparse
import os
import pickle
import torch
from models.encoders.lstm_opt_encoder import Encoder
from models.decoders.lstm_opt_decoder import Decoder
from models.seq2seq.seq2seq import SequenceToSequence
import warnings
from nltk.translate.bleu_score import sentence_bleu

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to a config file")
args = parser.parse_args()

config_path = args.config_path
with open(config_path, 'r') as read_file:
    config = yaml.safe_load(read_file)

data_path = config["raw_data_path"]
pickled_data_path = config["pickled_data_path"]
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
destination_path = config["destination_path"]
beam_width = config["beam_width"]

input_dir = data_path + "/" + "tokens_input"
output_dir = data_path + "/" + "tokens_output"
input_files = sorted(os.listdir(input_dir))

# take last 10k samples into test set
test_input_files = input_files[-10000:]

with open(pickled_data_path + "/" + "output_lang.pickle", 'rb') as read_file:
    output_lang = pickle.load(read_file)

with open(pickled_data_path + "/" + "input_lang.pickle", 'rb') as read_file:
    input_lang = pickle.load(read_file)

with open(pickled_data_path + "/" + "stats.pickle", 'rb') as read_file:
    stats = pickle.load(read_file)

max_length = stats["max_length"]
input_lang_unique_tokens = stats["input_lang_tokens"]
output_lang_unique_tokens = stats["output_lang_tokens"]

enc = Encoder(input_dim=input_lang_unique_tokens, emb_dim=input_embedding_dim,
              hid_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

dec = Decoder(output_dim=output_lang_unique_tokens, emb_dim=output_embedding_dim,
              hid_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

model = SequenceToSequence(encoder=enc, decoder=dec, device=device).to(device)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

# save the input_test_data and target_test_data for test dataset
print("running...")
bleu_scores = []
for file in test_input_files:
    target_file = file.strip().split("__")[0] + "--CAST.tcast"
    input_path = input_dir + "/" + file
    output_path = output_dir + "/" + target_file

    with open(input_path, 'r') as read_file:
        tokens = read_file.readline().strip()
        input_tensor, _ = input_lang.tensor_from_sequence(tokens)

    with open(output_path, 'r') as read_file:
        tokens = read_file.readline().strip()
        target_tensor, _ = output_lang.tensor_from_sequence(tokens)

    input_tensor = input_tensor.view(-1, 1).to(device)
    target_tensor = target_tensor.view(-1, 1).to(device)

    # decoded_output = model(input_tensor, target_tensor, test=True)
    decoded_outputs = model.beam_decode(input_tensor, target_tensor, beam_width=beam_width)
    target_seq = target_tensor.view(-1).cpu().tolist()
    target_tokens = [[output_lang.index2token[index] for index in target_seq]]
    outputs = []
    bleu = []
    for output_seq in decoded_outputs:
        output_seq = output_seq.split(",")
        output_seq = [int(x) for x in output_seq]
        try:
            eos_index = output_seq.index(3)
        except ValueError:
            eos_index = -1
        output_seq = output_seq[0: eos_index + 1]
        output_tokens = [output_lang.index2token[index] for index in output_seq]
        outputs.append(output_tokens)
        bleu_score = sentence_bleu(target_tokens, output_tokens)
        bleu.append(bleu_score)

    save_name = file.strip().split("__")[0] + '.txt'
    with open(destination_path + '/' + save_name, 'w') as write_file:
        write_file.write(f"input file: {file}\n")
        write_file.write(f"\ncast file: {target_file}\n")
        write_file.write(f"\ntarget sequence:\n{target_tokens}\n")
        for idx, output_tokens in enumerate(outputs):
            write_file.write(f"\nmodel output:\n{output_tokens}\n")
            write_file.write(f"\nbleu score: {bleu[idx]}\n")
        write_file.write(f"\nmax bleu score: {max(bleu)}\n")
        bleu_scores.append(max(bleu))
        write_file.write(f"\noutput with max bleu score:\n{outputs[bleu.index(max(bleu))]}")

print("Done writing files!")
print(f"avg bleu score using beam search: {round(sum(bleu_scores) / len(bleu_scores), 2)}")
