"""
generate bleu score for each of the test sample
generate txt file for each of the sample that has
target
output_from_model
bleu_score
sample config file: configs/test_bleu.yaml
use save_txt_files: false to just display the avg bleu score without saving the txt files
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
from nltk.translate.bleu_score import sentence_bleu
from utils.util import get_input_tokens_list_v3, get_output_tokens_list_v3
import numpy as np


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to a config file")
args = parser.parse_args()

config_path = args.config_path
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
save_txt_files = config["save_txt_files"]
max_length = config["max_length"]
copy_mechanism = config["copy_mechanism"]

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
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

# save the input_test_data and target_test_data for test dataset
print("running...")
bleu_scores = []
seq_lengths = []
exact_match_function = 0
exact_match_program = 0
total_functions = 0
total_programs = 0

for file in test_input_files:
    target_file = file.strip().split("__")[0] + "--CAST.tcast"
    input_path = input_dir + "/" + file
    output_path = output_dir + "/" + target_file

    input_tokens_dict, input_vals_dict, input_globals_dict = get_input_tokens_list_v3(input_path)
    output_tokens_dict, output_vals_dict, output_globals_dict = get_output_tokens_list_v3(output_path)

    if save_txt_files:
        # create a folder in destination path with the program name
        folder_name = file.strip().split("__")[0]
        folder_path = destination_path + os.sep + folder_name
        os.mkdir(folder_path)

    exact_match_info = []
    for func_name, token_seq_input in input_tokens_dict.items():
        input_tensor, _ = input_lang.tensor_from_sequence(token_seq_input)
        try:
            target_tokens = output_tokens_dict[func_name]
        except KeyError:
            raise Exception(f"Unable to find corresponding function name {func_name}\n\
                                of source file: {file}\n\
                                in target file: {target_file}")

        target_tensor, _ = output_lang.tensor_from_sequence(target_tokens)
        input_tensor = input_tensor.view(-1, 1).to(device)
        target_tensor = target_tensor.view(-1, 1).to(device)

        target_sos_token = 2  # fixed in our input and output language
        target_eos_token = 3  # fixed in our input and output language

        attention = None
        if model_type == "transformer":
            if copy_mechanism:
                input_tensor = input_tensor.to(device).T
                output_seq, attention = model.predict(input_tensor, target_sos_token,
                                                      target_eos_token, max_length=max_length,
                                                      return_attention_map=True)
                attention = attention.squeeze(0).cpu().numpy()
                attention = np.mean(attention, axis=0)
            else:
                output_seq = model.predict(input_tensor.T, target_sos_token,
                                           target_eos_token, max_length=max_length)
        else:
            output_seq = model.predict(input_tensor, target_sos_token,
                                       target_eos_token, max_length=max_length)

        target_seq = target_tensor.view(-1).cpu().tolist()
        try:
            eos_index = output_seq.index(3)
        except ValueError:
            eos_index = len(output_seq) - 1

        output_seq = output_seq[0: eos_index + 1]
        target_tokens = [output_lang.index2token[index] for index in target_seq]
        # modify target tokens to have actual values
        if copy_mechanism:
            # put actual values of "val#" from dictionary for output target tokens
            new_target_tokens = []
            for token in target_tokens:
                if "val" in token:
                    new_target_tokens.append(output_vals_dict[func_name][token])
                elif "_g" in token:
                    new_target_tokens.append(output_globals_dict[token])
                else:
                    new_target_tokens.append(token)
            target_tokens = new_target_tokens

        output_tokens = [output_lang.index2token[index] for index in output_seq]
        # if copy mechanism - replace the val# with actual token
        # modify predicted sequence to have actual values by copying from the input sequence
        if copy_mechanism:
            input_val_indices = []
            input_val_list = []
            input_global_indices = []
            input_global_list = []
            # token_seq_input doesnot have <sos> and <eos> tokens, but the input to the system has
            # these tokens: we should consider them while taking attention over the input sequence
            token_seq_input.insert(0, '<sos>')
            token_seq_input.append('<eos>')
            for index, token in enumerate(token_seq_input):
                if "_v" in token.lower():
                    input_val_indices.append(index)
                    input_val_list.append(token)
                elif "_g" in token.lower():
                    input_global_indices.append(index)
                    input_global_list.append(token)

            new_output_tokens = []
            for index, token in enumerate(output_tokens):
                if "val" in token.lower():
                    try:
                        attention_vector = attention[index - 1]
                        attention_over_input_val = attention_vector[input_val_indices]
                        selected_input_val = input_val_list[np.argmax(attention_over_input_val)]
                        new_output_tokens.append(input_vals_dict[func_name][selected_input_val])
                    except KeyError:
                        new_output_tokens.append(token)
                    except ValueError:
                        new_output_tokens.append(token)

                elif "_g" in token.lower():
                    try:
                        attention_vector = attention[index - 1]
                        attention_over_input_global = attention_vector[input_global_indices]
                        selected_global = input_global_list[np.argmax(attention_over_input_global)]
                        new_output_tokens.append(input_globals_dict[selected_global])
                    except KeyError:
                        new_output_tokens.append(token)
                    except ValueError:
                        new_output_tokens.append(token)

                else:
                    new_output_tokens.append(token)
            output_tokens = new_output_tokens

        target_tokens = [target_tokens]
        bleu_score = sentence_bleu(target_tokens, output_tokens)
        # some stats
        exact_match = (target_tokens[0] == output_tokens)

        if exact_match:
            exact_match_info.append(True)
        else:
            exact_match_info.append(False)

        seq_length = len(target_tokens[0])
        seq_lengths.append(seq_length)

        if save_txt_files:
            save_name_file = func_name + ".txt"
            with open(folder_path + os.sep + save_name_file, 'w') as write_file:
                write_file.write(f"input file: {file}\n")
                write_file.write(f"cast file: {target_file}\n")
                write_file.write(f"target sequence:\n{target_tokens}\n")
                write_file.write(f"model output:\n{output_tokens}\n")
                write_file.write(f"bleu score: {bleu_score}\n")
                write_file.write(f"seq_length: {seq_length}\n")
                if exact_match:
                    write_file.write(f"exact_match: true\n")
                else:
                    write_file.write(f"exact_match: false\n")

        bleu_scores.append(bleu_score)
        total_functions += 1

    if all(exact_match_info):
        exact_match_program += 1

    exact_match_function += sum(exact_match_info)
    total_programs += 1

exact_match_function_percentage = round((exact_match_function / total_functions), 2)
exact_match_program_percentage = round((exact_match_program / total_programs), 2)
print(f"avg bleu: {round(sum(bleu_scores) / len(bleu_scores), 2)}")
print(f"avg seq_length: {round(sum(seq_lengths) / len(seq_lengths), 2)}")
print(f"exact_match_function_percentage: {exact_match_function_percentage}")
print(f"exact_match_program_percentage: {exact_match_program_percentage}")
print("Done!")
