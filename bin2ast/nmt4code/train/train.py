"""
legacy training code: that works for batch_size = 1 only and works on
one word at a time for encoder and decoder: works good but takes a lot of time to train
"""

import yaml
import argparse
from models.encoders.lstm_encoder import LSTMEncoder
from models.decoders.lstm_decoder import LSTMDecoder
import torch
from torch import optim
from torch import nn
from nltk.translate.bleu_score import sentence_bleu
import pickle
import random
from tqdm import tqdm
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to a config file")
args = parser.parse_args()

config_path = args.config_path
with open(config_path, 'r') as read_file:
    config = yaml.safe_load(read_file)

gpu_index = config["gpu_index"]
device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]
pickled_data_path = config["pickled_data_path"]
hidden_dim = config["hidden_dim"]
print_every = config["print_every"]
model_path = config["model_path"]
start_from_pretrained = config["start_from_pretrained"]

# sos and eos tokens
sos_token = 0
eos_token = 1

print("loading training data...")
# load pickled files and print some stats
with open(pickled_data_path + "/" + "stats.pickle", 'rb') as read_file:
    stats = pickle.load(read_file)

max_length = stats["max_length"]
input_lang_unique_tokens = stats["input_lang_tokens"]
output_lang_unique_tokens = stats["output_lang_tokens"]

# load output lang: for decoding tokens from index to actual tokens
with open(pickled_data_path + "/" + "output_lang.pickle", 'rb') as read_file:
    output_lang = pickle.load(read_file)

# load train input data
with open(pickled_data_path + "/" + "train_input_data.pickle", 'rb') as read_file:
    train_input_data = pickle.load(read_file)

# load train target data
with open(pickled_data_path + "/" + "train_target_data.pickle", 'rb') as read_file:
    train_target_data = pickle.load(read_file)

# load val input data
with open(pickled_data_path + "/" + "val_input_data.pickle", 'rb') as read_file:
    val_input_data = pickle.load(read_file)

# load val target data
with open(pickled_data_path + "/" + "val_target_data.pickle", 'rb') as read_file:
    val_target_data = pickle.load(read_file)

# load test input data
with open(pickled_data_path + "/" + "test_input_data.pickle", 'rb') as read_file:
    test_input_data = pickle.load(read_file)

# load test target data
with open(pickled_data_path + "/" + "test_target_data.pickle", 'rb') as read_file:
    test_target_data = pickle.load(read_file)

# create encoder and decoder
if start_from_pretrained:
    encoder = torch.load(model_path + "/" + "encoder.pth")
    decoder = torch.load(model_path + "/" + "decoder.pth")
else:
    encoder = LSTMEncoder(input_size=input_lang_unique_tokens, hidden_size=hidden_dim).to(device)
    decoder = LSTMDecoder(hidden_size=hidden_dim, output_size=output_lang_unique_tokens).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

total_train_samples = len(train_input_data)
train_indices = list(range(total_train_samples))

total_val_samples = len(val_input_data)
total_test_samples = len(test_input_data)

best_val_bleu_score = 0
best_encoder = encoder
best_decoder = decoder
display_loss = []
counter = 0

for epoch in range(num_epochs):
    print(f"epoch: {epoch + 1} / {num_epochs}")
    random.shuffle(train_indices)
    for idx in tqdm(train_indices):
        encoder_hidden = encoder.init_hidden(device=device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        inp_tensor = train_input_data[idx].to(device)
        out_tensor = train_target_data[idx].to(device)

        inp_length = inp_tensor.shape[0]
        out_length = out_tensor.shape[0]

        for inp_idx in range(inp_length):
            encoder_output, encoder_hidden = encoder(inp_tensor[inp_idx], encoder_hidden)

        decoder_input = torch.tensor([[sos_token]]).to(device)
        decoder_hidden = encoder_hidden

        loss = 0.0
        for out_idx in range(out_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, out_tensor[out_idx])
            if decoder_input.item() == eos_token:
                break

        loss.backward()
        decoder_optimizer.step()
        encoder_optimizer.step()

        counter += 1
        display_loss.append(loss.item())
        if counter % print_every == 0:
            avg_loss = sum(display_loss) / len(display_loss)
            print(f"\tloss: {round(avg_loss, 2)}")
            display_loss = []

    # evaluate bleu scores on validation dataset
    bleu_scores = []
    with torch.no_grad():
        for idx in range(total_val_samples):
            encoder_hidden = encoder.init_hidden(device=device)

            inp_tensor = val_input_data[idx].to(device)
            out_tensor = val_target_data[idx].to(device)

            inp_length = inp_tensor.shape[0]
            out_length = out_tensor.shape[0]

            for inp_idx in range(inp_length):
                encoder_output, encoder_hidden = encoder(inp_tensor[inp_idx], encoder_hidden)

            decoder_input = torch.tensor([[sos_token]]).to(device)
            decoder_hidden = encoder_hidden

            decoded_tokens = []
            for _ in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                if topi.item() == eos_token:
                    decoded_tokens.append('eos')
                    break
                else:
                    decoded_tokens.append(output_lang.index2token[topi.item()])

                decoder_input = topi.squeeze().detach()

            reference = [[output_lang.index2token[seq_idx.item()] for seq_idx in out_tensor]]
            bleu_scores.append(sentence_bleu(reference, decoded_tokens))

    avg_score = round((sum(bleu_scores) / len(bleu_scores)), 2)
    print(f"Avg BLEU Score [val]: {avg_score}")

    if avg_score > best_val_bleu_score:
        best_encoder = deepcopy(encoder)
        best_decoder = deepcopy(decoder)
        best_val_bleu_score = avg_score

# save trained models
encoder_path = model_path + "/" + "encoder.pth"
torch.save(best_encoder, encoder_path)
decoder_path = model_path + "/" + "decoder.pth"
torch.save(best_decoder, decoder_path)

# evaluate on test samples
bleu_scores = []
# display 10 sample outputs
counter = 0
with torch.no_grad():
    for idx in range(total_test_samples):
        encoder_hidden = best_encoder.init_hidden(device=device)

        inp_tensor = test_input_data[idx].to(device)
        out_tensor = test_target_data[idx].to(device)

        inp_length = inp_tensor.shape[0]
        out_length = out_tensor.shape[0]

        for inp_idx in range(inp_length):
            encoder_output, encoder_hidden = best_encoder(inp_tensor[inp_idx], encoder_hidden)

        decoder_input = torch.tensor([[output_lang.sos_token]]).to(device)
        decoder_hidden = encoder_hidden

        decoded_tokens = []
        for _ in range(max_length):
            decoder_output, decoder_hidden = best_decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == output_lang.eos_token:
                decoded_tokens.append('eos')
                break
            else:
                decoded_tokens.append(output_lang.index2token[topi.item()])

            decoder_input = topi.squeeze().detach()

        reference = [[output_lang.index2token[seq_idx.item()] for seq_idx in out_tensor]]
        bleu_scores.append(sentence_bleu(reference, decoded_tokens))
        if counter % 1000 == 0:
            print(f"target: {reference}")
            print(f"output: {decoded_tokens}")
        counter += 1

avg_score = round((sum(bleu_scores) / len(bleu_scores)), 2)
print(f"Avg BLEU Score [test]: {avg_score}")
