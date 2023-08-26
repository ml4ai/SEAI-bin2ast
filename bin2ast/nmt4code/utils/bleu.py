"""
evaluate bleu score on the given (validation/test) dataset: evaluate_bleu
also return sample results: sample_result
"""

import torch
from nltk.translate.bleu_score import sentence_bleu


def evaluate_bleu(m, loader, output_lang, model_type, device):
    """
    evaluates bleu score on validation/test dataset
    :param m: model
    :param loader: validation/test loader
    :param output_lang: output language
    :param model_type: type of model
    :param device: cpu/gpu
    :return: average bleu score on the given dataset
    """

    m.eval()
    bleu_scores = []
    with torch.no_grad():
        for batch in loader:
            src = batch[0].to(device)
            target = batch[1].to(device)

            if model_type == "attention":
                src_lengths = batch[2]
                output = m(src, src_lengths, target)
                # decoded_output = some_processing(output)
                # need to check the following code: but we don't need to worry now
                decoded_output = output.argmax(dim=1)
            elif model_type == "transformer":
                src, target = src.T, target.T
                output, _ = m(src, target[:, :-1])
            else:
                output = m(src, target)
                # decoded_output = some_processing(output)
                # need to check the following code: but we don't need to worry now
                decoded_output = output.argmax(dim=1)

            if model_type == "transformer":
                reference = target
                output = output.argmax(dim=-1)
            else:
                output = torch.stack(decoded_output).T
                reference = target.T

            b_size, seq_len = reference.shape  # target will have the same shape
            for idx in range(b_size):
                target_seq = reference[idx].cpu().tolist()
                if model_type == "transformer":
                    output_seq = [target[0][0].item()] + output[idx].cpu().tolist()
                else:
                    output_seq = output[idx].cpu().tolist()
                # remove everything from <eos> token: 3
                target_seq = target_seq[0: target_seq.index(3) + 1]
                # handle no <eos> tokens in some output_seq
                try:
                    eos_index = output_seq.index(3)
                except ValueError:
                    eos_index = len(output_seq) - 1
                output_seq = output_seq[0: eos_index + 1]
                target_tokens = [[output_lang.index2token[index] for index in target_seq]]
                output_tokens = [output_lang.index2token[index] for index in output_seq]
                bleu_score = sentence_bleu(target_tokens, output_tokens)
                bleu_scores.append(bleu_score)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu


def sample_result(m, loader, output_lang, device, n=2):
    """
    returns sample results along with target from given dataset loader
    :param m: model
    :param loader: dataset loader (validation/test)
    :param output_lang: output language
    :param device: cpu/gpu
    :param n: number of samples to return
    :return: target, model_output pair sample results
    """

    m.eval()
    counter = 0
    result = []
    with torch.no_grad():
        for batch in loader:
            src = batch[0].to(device)
            target = batch[1].to(device)
            output = m(src, target)
            # decoded_output = some_processing(output)
            # need to check the following code: but we don't need to worry now
            decoded_output = output.argmax(dim=1)
            output = torch.stack(decoded_output).T
            reference = target.T
            b_size, seq_len = reference.shape  # target will have the same shape
            for idx in range(b_size):
                target_seq = reference[idx].cpu().tolist()
                output_seq = output[idx].cpu().tolist()
                # remove everything from <eos> token: 3
                target_seq = target_seq[0: target_seq.index(3) + 1]
                # handle no <eos> tokens in some output_seq
                try:
                    eos_index = output_seq.index(3)
                except ValueError:
                    eos_index = -1
                output_seq = output_seq[0: eos_index + 1]
                target_tokens = [[output_lang.index2token[index] for index in target_seq]]
                output_tokens = [output_lang.index2token[index] for index in output_seq]
                result.append([target_tokens, output_tokens])
                if counter == n:
                    break
                counter += 1
    return result
