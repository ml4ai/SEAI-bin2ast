"""
load the txt files generated from beam search and display the average bleu value
use the max bleu score from beam search
"""
import os

data_path = "/Users/kcdharma/results/local/nmt4code/beam_search"
files = os.listdir(data_path)

bleu_scores = []
print("running...")
for file in files:
    with open(data_path + "/" + file, 'r') as read_file:
        for line in read_file:
            if line.startswith("max bleu score:"):
                _, bleu = line.strip().split(":")
                bleu = float(bleu.strip())
                bleu_scores.append(bleu)

print(f"avg bleu score: {round(sum(bleu_scores) / len(bleu_scores), 2)}")
