"""
Generate the responsibility map for the output sequence wrt to input sequence
if save_all: true in yaml file: loop over all test examples
create a folder and save the responsibility map along with the predictions
else just show the responsibility map for one random example
"""

import yaml
import argparse
import os
import pickle
import torch
from models.encoders.lstm_opt_encoder import Encoder
from models.decoders.lstm_opt_decoder import Decoder
from models.seq2seq.seq2seq import SequenceToSequence
from models.attention.attention import Attention
from models.encoders.attention_encoder import AttentionEncoder
from models.decoders.attention_decoder import AttentionDecoder
from models.seq2seq.seq2seq_attention import SequenceToSequenceWithAttention
from models.encoders.transformer_encoder import TransformerEncoder
from models.decoders.transformer_decoder import TransformerDecoder
from models.seq2seq.seq2seq_transformer import SequenceToSequenceWithTransformer
import warnings
from utils.util import get_input_tokens_list, get_output_tokens_list
from utils.display import save_responsibility_map
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="path to a config file")
args = parser.parse_args()

config_path = args.config
with open(config_path, 'r') as read_file:
    config = yaml.safe_load(read_file)

data_path = config["raw_data_path"]
gpu_index = config["gpu_index"]
device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
pickled_data_path = config["pickled_data_path"]
enc_hidden_dim = config["enc_hidden_dim"]
dec_hidden_dim = config["dec_hidden_dim"]
model_path = config["model_path"]
input_embedding_dim = config["input_embedding_dim"]
output_embedding_dim = config["output_embedding_dim"]
n_layers = config["n_layers"]
dropout = config["dropout"]
destination_path = config["destination_path"]
model_type = config["model_type"]
save_all = config["save_all"]
max_length = config["max_length"]
top_k = config["top_k"]
k = config["k"]

# transformer parameters
transformer_hidden_dim = config["transformer_hidden_dim"]
transformer_n_heads = config["transformer_n_heads"]
transformer_enc_layers = config["transformer_enc_layers"]
transformer_dec_layers = config["transformer_dec_layers"]
transformer_pf_dim = config["transformer_pf_dim"]
transformer_enc_dropout = config["transformer_enc_dropout"]
transformer_dec_dropout = config["transformer_dec_dropout"]

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

input_lang_unique_tokens = stats["input_lang_tokens"]
output_lang_unique_tokens = stats["output_lang_tokens"]

if model_type == "attention":
    attn = Attention(enc_hidden_dim, dec_hidden_dim)
    enc = AttentionEncoder(input_dim=input_lang_unique_tokens, emb_dim=input_embedding_dim,
                           enc_hid_dim=enc_hidden_dim, dec_hid_dim=dec_hidden_dim, dropout=dropout)
    dec = AttentionDecoder(output_dim=output_lang_unique_tokens, emb_dim=output_embedding_dim,
                           enc_hid_dim=enc_hidden_dim, dec_hid_dim=dec_hidden_dim,
                           dropout=dropout, attention=attn)
    model = SequenceToSequenceWithAttention(enc, dec, device).to(device)

elif model_type == "transformer":
    enc = TransformerEncoder(inp_dim=input_lang_unique_tokens, hid_dim=transformer_hidden_dim,
                             n_layers=transformer_enc_layers, n_heads=transformer_n_heads,
                             pf_dim=transformer_pf_dim, dropout=transformer_enc_dropout,
                             devce=device, max_length=max_length)
    dec = TransformerDecoder(output_dim=output_lang_unique_tokens, hid_dim=transformer_hidden_dim,
                             n_layers=transformer_dec_layers, n_heads=transformer_n_heads,
                             pf_dim=transformer_pf_dim, dropout=transformer_dec_dropout,
                             devc=device, max_length=max_length)
    model = SequenceToSequenceWithTransformer(encoder=enc, decoder=dec, src_pad_idx=0,
                                              trg_pad_idx=0, devc=device).to(device)
else:
    enc = Encoder(input_dim=input_lang_unique_tokens, emb_dim=input_embedding_dim,
                  hid_dim=enc_hidden_dim, n_layers=n_layers, dropout=dropout)
    dec = Decoder(output_dim=output_lang_unique_tokens, emb_dim=output_embedding_dim,
                  hid_dim=dec_hidden_dim, n_layers=n_layers, dropout=dropout)
    model = SequenceToSequence(encoder=enc, decoder=dec, device=device).to(device)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

# save the input_test_data and target_test_data for test dataset
print("running...")

for file in test_input_files:
    target_file = file.strip().split("__")[0] + "--CAST.tcast"
    input_path = input_dir + "/" + file
    output_path = output_dir + "/" + target_file

    input_tokens_list = get_input_tokens_list(input_path, with_values=False,
                                              fix_hex_values=False)
    input_tensor, _ = input_lang.tensor_from_sequence(input_tokens_list)

    output_tokens_list = get_output_tokens_list(output_path, with_values=False)
    target_tensor, _ = output_lang.tensor_from_sequence(output_tokens_list)

    input_tensor = input_tensor.view(-1, 1).to(device)
    target_tensor = target_tensor.view(-1, 1).to(device)

    target_sos_token = 2  # fixed in our input and output language
    target_eos_token = 3  # fixed in our input and output language

    if model_type == "transformer":
        output_seq, attention = model.predict(input_tensor.T, target_sos_token,
                                              target_eos_token, max_length=max_length,
                                              return_attention_map=True)
    else:
        output_seq, attention = model.predict(input_tensor, target_sos_token,
                                              target_eos_token, max_length=max_length)

    target_seq = target_tensor.view(-1).cpu().tolist()
    try:
        eos_index = output_seq.index(3)
    except ValueError:
        eos_index = len(output_seq) - 1

    output_seq = output_seq[0: eos_index + 1]
    target_tokens = [output_lang.index2token[index] for index in target_seq]
    output_tokens = [output_lang.index2token[index] for index in output_seq]

    save_folder = file.strip().split("__")[0]
    save_folder_path = Path(destination_path + "/" + save_folder)
    save_folder_path.mkdir(parents=True, exist_ok=True)

    save_name = save_folder + '.txt'
    with open(save_folder_path / save_name, 'w') as write_file:
        write_file.write(f"input file: {file}\n")
        write_file.write(f"\ncast file: {target_file}\n")
        write_file.write(f"\ninput sequence: {input_tokens_list}\n")
        write_file.write(f"\ntarget sequence:\n{target_tokens}\n")
        write_file.write(f"\nmodel output:\n{output_tokens}\n")
        # remove <sos> from output sequence for further analysis: it's not the
        # output token generated by the decoder but is the token to start
        output_tokens = output_tokens[1:]
        # write the top k values of responsibility for the given decoded token
        if top_k:
            _attention = attention.squeeze(0).cpu().detach().numpy()
            _attention = np.mean(_attention, axis=0)
            input_tensor_reshaped = input_tensor.reshape(-1).numpy()
            for output_index, output_token in enumerate(output_tokens):
                write_file.write(f"\noutput_index: {output_index}, output_token: {output_token}\n")
                _attention_vector = _attention[output_index]
                top_indices_unsorted = np.argpartition(_attention_vector, -k)[-k:]
                top_indices_sorted = top_indices_unsorted[np.argsort(
                    _attention_vector[top_indices_unsorted])][::-1]
                top_responsibilities = _attention_vector[top_indices_sorted].tolist()
                top_token_indices = input_tensor_reshaped[top_indices_sorted].tolist()
                top_tokens = [input_lang.index2token[idx] for idx in top_token_indices]
                for index, token, responsibility in zip(top_token_indices, top_tokens, top_responsibilities):
                    write_file.write(f"\tinput_index: {index}, input_token: {token},"
                                     f"responsibility: {responsibility}\n")
    # save the responsibility map
    save_responsibility_map(input_tokens_list, output_tokens, attention, save_folder_path)
    if not save_all:
        break

print("Done!")
