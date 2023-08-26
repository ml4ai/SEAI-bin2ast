"""
calculate the bleu score of the results from v3 experiment
remove occurance of _f [functions being called] from the prediction
of given function
"""

import os
from nltk.translate.bleu_score import sentence_bleu
import re
import ast

# parent directory that consists of list of directories
v3_result_folder = "/home/kcdharma/results/nmt4code/test_bleu/v3/transformer"

folders = os.listdir(v3_result_folder)

bleu_scores = []
exact_match_count = 0.0
total = 0

print("Running...")

for folder in folders:
    folder_path = v3_result_folder + os.sep + folder
    txt_files = os.listdir(folder_path)
    for txt_file in txt_files:
        with open(folder_path + os.sep + txt_file, 'r') as read_file:
            for line in read_file:
                if line.startswith("target sequence"):
                    target_seq = ast.literal_eval(next(read_file).strip())
                if line.startswith("model output"):
                    pred_seq = ast.literal_eval(next(read_file).strip())

        # remove tokens with _f in their names: functions being called from target sequence
        for index, item in enumerate(target_seq[0]):
            if re.search("^_f[0-9]+$", item):
                target_seq[0][index] = "_f"

        # remove tokens with _f in their names: functions being alled from predicted sequence
        for index, item in enumerate(pred_seq):
            if re.search("^_f[0-9]+$", item):
                pred_seq[index] = "_f"

        bleu_score = sentence_bleu(target_seq, pred_seq)

        # some stats
        exact_match = (target_seq[0] == pred_seq)

        bleu_scores.append(bleu_score)
        total += 1
        if exact_match:
            exact_match_count += 1

print(f"avg bleu: {round(sum(bleu_scores) / len(bleu_scores), 2)}")
exact_match_percentage = round((exact_match_count / total), 2)
print(f"exact_match_percentage: {exact_match_percentage}")
print("Done!")
