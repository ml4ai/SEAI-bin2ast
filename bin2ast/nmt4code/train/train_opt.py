"""
Train the optimized code that can handle multiple sequences at a time: batch_size > 1
train_opt: train the optimized version
"""

import matplotlib.pyplot as plt
import yaml
import argparse
import torch
from data.pickle_loader import load_pickled_data
from data.data_loader import get_loaders
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
from models.decoders.tree_decoder import TreeDecoder
from models.seq2tree.seq2tree import TransformerEncoderTreeDecoder
import time
from torch import optim
from torch import nn
import math
import warnings
from train_epoch import train_epoch
from evaluate_epoch import evaluate_epoch
from utils.util import epoch_time, init_weights, init_weights_attn, init_weights_transformer
from utils.util import init_weights_tree_decoder, init_weights_gnn_decoder
from utils.ddp import get_free_port, setup_ddp
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from models.seq2tree.seq2treegnn import TransformerEncoderGnnDecoder
from models.decoders.gnn_decoder import GnnDecoder


warnings.filterwarnings('ignore')


def get_loaders_and_stats(config):
    """
    load the pickled dataset, get the loaders and return them
    also return some stats along with output lang
    """
    return_list = list()
    pickled_data_path = config["pickled_data_path"]
    pickled_data = load_pickled_data(pickled_data_path=pickled_data_path)
    train_input_data = pickled_data['train_input_data']
    train_target_data = pickled_data['train_target_data']
    val_input_data = pickled_data['val_input_data']
    val_target_data = pickled_data['val_target_data']
    test_input_data = pickled_data['test_input_data']
    test_target_data = pickled_data['test_target_data']
    input_lang = pickled_data["input_lang"]
    batch_size = config["batch_size"]
    data_augmentation = config["data_augmentation"]
    stats = pickled_data['stats']
    input_lang_unique_tokens = stats["input_lang_tokens"]
    output_lang_unique_tokens = stats["output_lang_tokens"]
    output_lang = pickled_data['output_lang']
    is_distributed = config["distributed"]

    # train loader uses distributed sampler for ddp training if is_distributed is set to true
    # validation loader and test loader do not use ddp for now: just evaluate using a single gpu
    # because it is not that computationally expensive
    train_loader, val_loader, test_loader = get_loaders(train_input_data,
                                                        train_target_data,
                                                        val_input_data,
                                                        val_target_data,
                                                        test_input_data,
                                                        test_target_data,
                                                        input_lang=input_lang,
                                                        data_augmentation=data_augmentation,
                                                        batch_size=batch_size,
                                                        is_distributed=is_distributed)
    return_list.extend([train_loader, val_loader, test_loader])  # loaders
    return_list.extend([input_lang_unique_tokens, output_lang_unique_tokens, output_lang])  # stats
    return return_list


def get_model_and_optimizer(config, inp_unique_tokens, out_unique_tokens, output_lang, world_size):
    enc_hidden_dim = config["enc_hidden_dim"]
    dec_hidden_dim = config["dec_hidden_dim"]
    input_embedding_dim = config["input_embedding_dim"]
    output_embedding_dim = config["output_embedding_dim"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    model_type = config["model_type"]
    learning_rate = config["learning_rate"]

    # transformer parameters
    transformer_hidden_dim = config["transformer_hidden_dim"]
    transformer_n_heads = config["transformer_n_heads"]
    transformer_enc_layers = config["transformer_enc_layers"]
    transformer_dec_layers = config["transformer_dec_layers"]
    transformer_pf_dim = config["transformer_pf_dim"]
    transformer_enc_dropout = config["transformer_enc_dropout"]
    transformer_dec_dropout = config["transformer_dec_dropout"]
    transformer_max_length = config["transformer_max_length"]
    transformer_lr = config["transformer_lr"]

    # parameters for tree decoder
    num_layers_encoder = config["num_layers_encoder"]
    parent_feeding = config["parent_feeding"]
    input_type = config["input_type"]
    max_nodes = config["max_nodes"]
    tree_decoder_lr = config["tree_decoder_lr"]
    tree_decoder_hidden_state_dim = config["tree_decoder_hidden_state_dim"]
    tree_decoder_embedding_dim = config["tree_decoder_embedding_dim"]
    use_teacher_forcing = config["use_teacher_forcing"]

    # parameters for transformer encoder and gnn decoder
    gnn_h_dim = config["gnn_h_dim"]
    gnn_num_layers_encoder = config["gnn_num_layers_encoder"]
    gnn_num_propagation = config["gnn_num_propagation"]
    gnn_lr = config["gnn_lr"]
    gnn_max_nodes = config["gnn_max_nodes"]
    gnn_n_heads = config["gnn_n_heads"]
    gnn_decoder_embedding_dim = config["gnn_decoder_embedding_dim"]

    if model_type == "attention":
        attn = Attention(enc_hidden_dim, dec_hidden_dim)
        enc = AttentionEncoder(input_dim=inp_unique_tokens, emb_dim=input_embedding_dim,
                               enc_hid_dim=enc_hidden_dim, dec_hid_dim=dec_hidden_dim, dropout=dropout)
        dec = AttentionDecoder(output_dim=out_unique_tokens, emb_dim=output_embedding_dim,
                               enc_hid_dim=enc_hidden_dim, dec_hid_dim=dec_hidden_dim,
                               dropout=dropout, attention=attn)
        model = SequenceToSequenceWithAttention(enc, dec)
        model.apply(init_weights_attn)

    elif model_type == "transformer":
        enc = TransformerEncoder(inp_dim=inp_unique_tokens, hid_dim=transformer_hidden_dim,
                                 n_layers=transformer_enc_layers, n_heads=transformer_n_heads,
                                 pf_dim=transformer_pf_dim, dropout=transformer_enc_dropout,
                                 max_length=transformer_max_length)
        dec = TransformerDecoder(output_dim=out_unique_tokens, hid_dim=transformer_hidden_dim,
                                 n_layers=transformer_dec_layers, n_heads=transformer_n_heads,
                                 pf_dim=transformer_pf_dim, dropout=transformer_dec_dropout,
                                 max_length=transformer_max_length)
        model = SequenceToSequenceWithTransformer(encoder=enc, decoder=dec, src_pad_idx=0,
                                                  trg_pad_idx=0)
        model.apply(init_weights_transformer)

    elif model_type == "tree_decoder":
        # add <eoc> token to output_lang for tree generative models
        output_lang.add_eoc()
        embedder = nn.Embedding(num_embeddings=inp_unique_tokens,
                                embedding_dim=tree_decoder_embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=tree_decoder_hidden_state_dim,
                                                   nhead=transformer_n_heads,
                                                   batch_first=True)
        enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers_encoder)
        # use our transformer encoder
        # enc = TransformerEncoder(inp_dim=inp_unique_tokens, hid_dim=transformer_hidden_dim,
        #                          n_layers=num_layers_encoder, n_heads=transformer_n_heads,
        #                          pf_dim=transformer_pf_dim, dropout=transformer_enc_dropout,
        #                          max_length=transformer_max_length)

        dec = TreeDecoder(h_dim=tree_decoder_hidden_state_dim,
                          output_vocab=output_lang.token2index,
                          parent_feeding=parent_feeding,
                          input_type=input_type,
                          max_nodes=max_nodes,
                          use_teacher_forcing=use_teacher_forcing)

        model = TransformerEncoderTreeDecoder(embedder=embedder, encoder=enc, decoder=dec, src_pad_idx=0)
        # model = TransformerEncoderTreeDecoder(encoder=enc, decoder=dec, src_pad_idx=0)
        model.apply(init_weights_tree_decoder)

    elif model_type == "gnn_decoder":
        # add <eoc> token to output_lang for tree generative models
        output_lang.add_eoc()
        embedder = nn.Embedding(num_embeddings=inp_unique_tokens,
                                embedding_dim=gnn_decoder_embedding_dim)

        # enc = TransformerEncoder(inp_dim=inp_unique_tokens, hid_dim=gnn_h_dim,
        #                          n_layers=gnn_num_layers_encoder, n_heads=gnn_n_heads,
        #                          pf_dim=transformer_pf_dim, dropout=transformer_enc_dropout,
        #                          max_length=transformer_max_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=gnn_h_dim,
                                                   nhead=gnn_n_heads,
                                                   batch_first=True)

        enc = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                    num_layers=gnn_num_layers_encoder)

        dec = GnnDecoder(num_propagation=gnn_num_propagation,
                         h_dim=gnn_h_dim,
                         out_vocab=output_lang.token2index,
                         max_nodes=gnn_max_nodes,
                         use_teacher_forcing=use_teacher_forcing)

        model = TransformerEncoderGnnDecoder(embedder=embedder, encoder=enc, decoder=dec, src_pad_idx=0)
        model.apply(init_weights_gnn_decoder)

    else:
        enc = Encoder(input_dim=inp_unique_tokens, emb_dim=input_embedding_dim,
                      hid_dim=enc_hidden_dim, n_layers=n_layers, dropout=dropout)
        dec = Decoder(output_dim=out_unique_tokens, emb_dim=output_embedding_dim,
                      hid_dim=dec_hidden_dim, n_layers=n_layers, dropout=dropout)
        model = SequenceToSequence(encoder=enc, decoder=dec)
        model.apply(init_weights)

    is_distributed = config["distributed"]
    if is_distributed:
        # modify the learning rate according to world_size: because the effective
        # batch size has increased
        transformer_lr = (world_size // 3) * transformer_lr
        learning_rate = (world_size // 3) * learning_rate
        tree_decoder_lr = (world_size // 3) * tree_decoder_lr
        gnn_lr = (world_size // 3) * gnn_lr

    if model_type == "transformer":
        optimizer = optim.Adam(model.parameters(), lr=transformer_lr)
    elif model_type == "tree_decoder":
        optimizer = optim.Adam(model.parameters(), lr=tree_decoder_lr)
    elif model_type == "gnn_decoder":
        optimizer = optim.Adam(model.parameters(), lr=gnn_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer


def plot_and_save(train_loss_list, val_loss_list, test_loss, plot_save_path, model_type):
    """
    display some stats and plot the losses
    """
    try:
        test_ppl = math.exp(test_loss)
    except OverflowError:
        test_ppl = float('inf')
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} |')
    # plot the figures and save the plots
    plt.figure(figsize=(10, 5))
    plt.title("Train/Val loss and val BLEU scores")
    plt.plot(train_loss_list, label="train_loss")
    plt.plot(val_loss_list, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss/bleu")
    plt.legend()
    plot_save_path = plot_save_path + "/" + model_type + ".png"
    plt.savefig(plot_save_path)


def _print(is_distributed, rank, print_string):
    """
    print something only on rank 0 if its is distributed training
    else print whatever the input is
    """
    if is_distributed:
        if rank == 0:
            print(print_string)
    else:
        print(print_string)


def train_evaluate(config, rank, model, optimizer, output_lang, train_loader,
                   val_loader, test_loader):
    pad_index = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    model_save_path = config["model_path"]
    model_type = config["model_type"]
    num_epochs = config["num_epochs"]
    clip = config["clip"]
    plot_save_path = config["plot_path"]
    is_distributed = config["distributed"]
    _print(is_distributed, rank, 'Training...')
    print_loss_minibatch = config["print_loss_minibatch"]

    orig_model = model  # save a copy to load from a saved location
    model = model.to(rank)
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # plot some statistics
    train_loss_list = []
    val_loss_list = []

    if is_distributed:
        model_save_path = model_save_path + '/' + 'best-model-' + model_type + '-dist.pt'
    else:
        model_save_path = model_save_path + '/' + 'best-model-' + model_type + '.pt'

    best_valid_loss = float('inf')
    val_loss, test_loss = None, None

    for epoch in range(num_epochs):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip, model_type,
                                 output_lang, rank, epoch, is_distributed, print_loss_minibatch)
        if is_distributed:
            if rank == 0:
                val_loss = evaluate_epoch(model, val_loader, criterion, model_type,
                                          output_lang, rank, epoch)
        else:
            val_loss = evaluate_epoch(model, val_loader, criterion, model_type,
                                      output_lang, rank, epoch)

        # if model_type == "tree_decoder" or model_type == "gnn_decoder":
        #     if model.decoder.use_teacher_forcing:
        #         model.decoder.teacher_forcing_ratio = round(math.exp(-epoch), 2)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if is_distributed:
            if rank == 0:
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    torch.save(model.module.state_dict(), model_save_path)
                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                # sometimes the loss is so high: model doesn't converge and math.exp(large_number)
                # throws error: guard against it
                try:
                    train_ppl = math.exp(train_loss)
                except OverflowError:
                    train_ppl = float('inf')
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
                try:
                    val_ppl = math.exp(val_loss)
                except OverflowError:
                    val_ppl = float('inf')
                print(f'\tVal Loss: {val_loss:.3f} |  Val. PPL: {val_ppl:7.3f}')
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
        else:
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            # sometimes the loss is so high: model doesn't converge and math.exp(large_number)
            # throws error: guard against it
            try:
                train_ppl = math.exp(train_loss)
            except OverflowError:
                train_ppl = float('inf')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
            try:
                val_ppl = math.exp(val_loss)
            except OverflowError:
                val_ppl = float('inf')
            print(f'\tVal Loss: {val_loss:.3f} |  Val. PPL: {val_ppl:7.3f}')
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

    if is_distributed:
        if rank == 0:
            device = torch.device("cuda:" + str(rank))
            orig_model.load_state_dict(torch.load(model_save_path,
                                                  map_location=device))
            test_loss = evaluate_epoch(orig_model, test_loader, criterion, model_type,
                                       output_lang, rank, epoch_num=num_epochs)
            plot_and_save(train_loss_list, val_loss_list, test_loss, plot_save_path, model_type)
    else:
        model.load_state_dict(torch.load(model_save_path))
        test_loss = evaluate_epoch(model, test_loader, criterion, model_type,
                                   output_lang, rank, epoch_num=num_epochs)
        plot_and_save(train_loss_list, val_loss_list, test_loss, plot_save_path, model_type)


def main(rank, world_size, f_port, config):
    is_distributed = config["distributed"]
    if is_distributed:
        setup_ddp(rank=rank, world_size=world_size, free_port=f_port)

    tn_ldr, vl_ldr, tt_ldr, inp_unq, out_unq, out_lang = get_loaders_and_stats(config=config)
    model, optimizer = get_model_and_optimizer(config, inp_unq, out_unq, out_lang, world_size)
    train_evaluate(config=config, model=model, optimizer=optimizer,
                   train_loader=tn_ldr, val_loader=vl_ldr,
                   test_loader=tt_ldr,
                   output_lang=out_lang, rank=rank)

    if is_distributed:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to a config file")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as read_file:
        cfg = yaml.safe_load(read_file)

    distributed = cfg["distributed"]
    m_type = cfg["model_type"]
    print(f"model_type: {m_type}")

    if distributed:
        w_size = torch.cuda.device_count()  # w_size: world size
        f_p = get_free_port()  # get free port
        print(f"using {w_size} gpus for distributed training!")
        mp.spawn(main, nprocs=w_size, args=(w_size, f_p, cfg))
    else:
        rnk = cfg["gpu_index"]
        print(f"using gpu: {rnk}")
        main(rank=rnk, world_size=1, f_port=None, config=cfg)
