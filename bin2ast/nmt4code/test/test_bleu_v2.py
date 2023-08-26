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
from utils.util import get_input_tokens_list_v2, get_output_tokens_list_v2
import numpy as np
from utils.util import is_exact_match_possible
from models.seq2tree.seq2tree import TransformerEncoderTreeDecoder
from models.decoders.tree_decoder import TreeDecoder
from utils.tree import nary_to_sequence

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
batch_size = config["batch_size"]
input_embedding_dim = config["input_embedding_dim"]
output_embedding_dim = config["output_embedding_dim"]
n_layers = config["n_layers"]
dropout = config["dropout"]
destination_path = config["destination_path"]
model_type = config["model_type"]
save_txt_files = config["save_txt_files"]
max_length = config["max_length"]
with_values = config["with_values"]
fix_hex_values = config["fix_hex_values"]
copy_mechanism = config["copy_mechanism"]
aggregate_attention = config["aggregate_attention"]
head_index = config["head_index"]

# transformer parameters
transformer_hidden_dim = config["transformer_hidden_dim"]
transformer_n_heads = config["transformer_n_heads"]
transformer_enc_layers = config["transformer_enc_layers"]
transformer_dec_layers = config["transformer_dec_layers"]
transformer_pf_dim = config["transformer_pf_dim"]
transformer_enc_dropout = config["transformer_enc_dropout"]
transformer_dec_dropout = config["transformer_dec_dropout"]

# parameters for tree decoder
num_layers_encoder = config["num_layers_encoder"]
parent_feeding = config["parent_feeding"]
input_type = config["input_type"]
max_nodes = config["max_nodes"]
tree_decoder_hidden_state_dim = config["tree_decoder_hidden_state_dim"]
tree_decoder_embedding_dim = config["tree_decoder_embedding_dim"]

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
    model = SequenceToSequenceWithAttention(enc, dec).to(device)

elif model_type == "transformer":
    enc = TransformerEncoder(inp_dim=input_lang_unique_tokens, hid_dim=transformer_hidden_dim,
                             n_layers=transformer_enc_layers, n_heads=transformer_n_heads,
                             pf_dim=transformer_pf_dim, dropout=transformer_enc_dropout,
                             max_length=max_length)
    dec = TransformerDecoder(output_dim=output_lang_unique_tokens, hid_dim=transformer_hidden_dim,
                             n_layers=transformer_dec_layers, n_heads=transformer_n_heads,
                             pf_dim=transformer_pf_dim, dropout=transformer_dec_dropout,
                             max_length=max_length)
    model = SequenceToSequenceWithTransformer(encoder=enc, decoder=dec, src_pad_idx=0,
                                              trg_pad_idx=0).to(device)
elif model_type == "tree_decoder":
    # add <eoc> token to output_lang for tree generative models
    output_lang.add_eoc()
    # embedder = nn.Embedding(num_embeddings=input_lang_unique_tokens,
    #                         embedding_dim=tree_decoder_embedding_dim)
    #
    # encoder_layer = nn.TransformerEncoderLayer(d_model=tree_decoder_hidden_state_dim,
    #                                            nhead=transformer_n_heads,
    #                                            batch_first=True)
    # enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers_encoder)

    enc = TransformerEncoder(inp_dim=input_lang_unique_tokens, hid_dim=transformer_hidden_dim,
                             n_layers=num_layers_encoder, n_heads=transformer_n_heads,
                             pf_dim=transformer_pf_dim, dropout=transformer_enc_dropout,
                             max_length=max_length)

    dec = TreeDecoder(h_dim=tree_decoder_hidden_state_dim,
                      output_vocab=output_lang.token2index,
                      parent_feeding=parent_feeding,
                      input_type=input_type,
                      max_nodes=max_nodes)

    # model = TransformerEncoderTreeDecoder(embedder=embedder, encoder=enc, decoder=dec).to(device)
    model = TransformerEncoderTreeDecoder(encoder=enc, decoder=dec, src_pad_idx=0).to(device)

else:
    enc = Encoder(input_dim=input_lang_unique_tokens, emb_dim=input_embedding_dim,
                  hid_dim=enc_hidden_dim, n_layers=n_layers, dropout=dropout)
    dec = Decoder(output_dim=output_lang_unique_tokens, emb_dim=output_embedding_dim,
                  hid_dim=dec_hidden_dim, n_layers=n_layers, dropout=dropout)
    model = SequenceToSequence(encoder=enc, decoder=dec).to(device)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

# save the input_test_data and target_test_data for test dataset
print("running...")
bleu_scores = []
seq_lengths = []
exact_match_count = 0
total = 0
total_values = 0
correct_values = 0
# find the maximum value of exact_match possible: due to information lost in Ghidra
# and tokenization
exact_match_possible_count = 0

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

    input_tensor, _ = input_lang.tensor_from_sequence(input_tokens_list)

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

    target_tensor, _ = output_lang.tensor_from_sequence(output_tokens_list)

    exact_match_possible = is_exact_match_possible(input_val_dict, output_val_dict)
    if exact_match_possible:
        exact_match_possible_count += 1

    input_tensor = input_tensor.view(-1, 1).to(device)
    target_tensor = target_tensor.view(-1, 1).to(device)

    target_sos_token = 2  # fixed in our input and output language
    target_eos_token = 3  # fixed in our input and output language

    attention = None
    # predicted val dict: compare with output_val_dict
    # figure out which of them match with which one
    predicted_output_val_dict = dict()
    if model_type == "transformer":
        if copy_mechanism:
            input_tensor = input_tensor.to(device).T
            output_seq, attention = model.predict(input_tensor, target_sos_token,
                                                  target_eos_token, max_length=max_length,
                                                  return_attention_map=True)
            attention = attention.squeeze(0).cpu().numpy()
            if head_index:
                attention = attention[head_index]
            else:
                attention = np.mean(attention, axis=0)
        else:
            output_seq = model.predict(input_tensor.T, target_sos_token,
                                       target_eos_token, max_length=max_length)
    elif model_type == "tree_decoder":
        loss = 0.0
        with torch.no_grad():
            input_tensor = input_tensor.to(device).T
            out_trees = model.predict(input_tensor)
            out_tree = out_trees[0]
            # while generating a tree: the method will not add child_val when it can not expand
            # it happens when the decoder can not extend the current node becasue of the node limits
            # in such case: modify the None nodes to <eoc> nodes
            out_tree.none_to_eoc()
            nary_tree = out_tree.get_nary_tree()
            output_seq = nary_to_sequence(nary_tree)
    else:
        output_seq = model.predict(input_tensor, target_sos_token,
                                   target_eos_token, max_length=max_length)

    target_seq = target_tensor.view(-1).cpu().tolist()
    if model_type == "tree_decoder":
        output_tokens = output_seq
    else:
        try:
            eos_index = output_seq.index(3)
        except ValueError:
            eos_index = len(output_seq) - 1
        output_seq = output_seq[0: eos_index + 1]
        output_tokens = [output_lang.index2token[index] for index in output_seq]

    target_tokens = [output_lang.index2token[index] for index in target_seq]
    # modify target sequence to have actual values
    if copy_mechanism:
        # put actual values of "val#" from dictionary for output target tokens
        new_target_tokens = []
        for token in target_tokens:
            if "val" in token.lower():
                new_target_tokens.append(output_val_dict[token])
            else:
                new_target_tokens.append(token)
        target_tokens = new_target_tokens

    # if copy mechanism - replace the val# with actual token
    if copy_mechanism:
        input_val_indices = []
        input_val_list = []
        # token_seq_input doesnot have <sos> and <eos> tokens, but the input to the system has
        # these tokens: we should consider them while taking attention over the input sequence
        input_tokens_list.insert(0, '<sos>')
        input_tokens_list.append('<eos>')
        for index, token in enumerate(input_tokens_list):
            if "_v" in token:
                input_val_indices.append(index)
                input_val_list.append(token)

        new_output_tokens = []
        for index, token in enumerate(output_tokens):
            if "val" in token.lower():
                try:
                    attention_vector = attention[index - 1]
                    attention_over_input_val = attention_vector[input_val_indices]
                    if aggregate_attention:
                        agg_attention = dict()
                        for idx, attn in enumerate(attention_over_input_val):
                            val = input_val_list[idx]
                            if val in agg_attention:
                                agg_attention[val] += attn
                            else:
                                agg_attention[val] = attn
                        selected_input_val = max(agg_attention, key=agg_attention.get)
                    else:
                        selected_input_val = input_val_list[np.argmax(attention_over_input_val)]

                    actual_value = input_val_dict[selected_input_val.strip()]
                    new_output_tokens.append(actual_value)
                    predicted_output_val_dict[token] = actual_value
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
        exact_match_count += 1

    seq_length = len(target_tokens[0])
    seq_lengths.append(seq_length)

    for key, val in output_val_dict.items():
        predicted_value = predicted_output_val_dict.get(key, None)
        if predicted_value == val:
            correct_values += 1
        total_values += 1

    if save_txt_files:
        save_name = file.strip().split("__")[0] + '.txt'
        with open(destination_path + '/' + save_name, 'w') as write_file:
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
    total += 1

exact_match_percentage = round((exact_match_count / total), 2)
correct_values_percentage = round((correct_values / total_values), 2)
print(f"avg bleu: {round(sum(bleu_scores) / len(bleu_scores), 2)}")
print(f"avg seq_length: {round(sum(seq_lengths) / len(seq_lengths), 2)}")
print(f"exact_match: {exact_match_percentage}")
print(f"total_values: {total_values}")
print(f"correctly predicted values: {correct_values}")
print(f"correct_values_percentagee: {correct_values_percentage}")
exact_match_possible_percentage = round((exact_match_possible_count / total), 2)
print(f"maximum possible value of exact match: {exact_match_possible_percentage}")
print("Done!")
