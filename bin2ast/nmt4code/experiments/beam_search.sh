# generate bleu score for each of the test sample
# generate txt file for each of the sample that has
# target
# output_from_model
# bleu_score
# upto beam_width times and
# also store the max bleu score among these bleu scores

python3 test/beam_search.py --config configs/beam_search.yaml
