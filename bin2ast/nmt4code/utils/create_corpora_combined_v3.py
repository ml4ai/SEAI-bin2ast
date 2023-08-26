# loop over corpora_1, corpora_2, corpora_3, corpora_4, corpora_5
# find input and output pairs that match the successfully generated samples from
# log_progress_*.txt and copy only those to tokens_input and tokens_output

import os
import shutil

corpora_combined_path = "corpora_combined"
input_directory = corpora_combined_path + os.sep + "tokens_input"
output_directory = corpora_combined_path + os.sep + "tokens_output"

if os.path.isdir(corpora_combined_path):
    # remove previous
    shutil.rmtree(corpora_combined_path)

# create new
os.mkdir(corpora_combined_path)
os.mkdir(input_directory)
os.mkdir(output_directory)

corporas = ["corpora_1", "corpora_2", "corpora_3", "corpora_4", "corpora_5"]
print("creating corpora_combined")

for corpora in corporas:
    corpora_number = corpora.split("_")[-1]
    log_file = "log_progress_" + corpora_number + ".txt"
    successfull_files = []
    with open(log_file) as read_file:
        for line in read_file:
            successfull_files.append(line.strip())

    corpus_folders = next(os.walk(corpora))[1]
    for corpus_folder in corpus_folders:
        token_input_folder = corpora + os.sep + corpus_folder + os.sep + "tokens_input"
        token_output_folder = corpora + os.sep + corpus_folder + os.sep + "tokens_output"
        token_input_files = os.listdir(token_input_folder)
        for token_input_file in token_input_files:
            input_file_name = token_input_file.split("__")[0]
            file_number = input_file_name.split("_")[-1]
            output_file_name = input_file_name + '--CAST.tcast'
            if file_number in successfull_files:
                input_src_file = token_input_folder + os.sep + token_input_file
                output_src_file = token_output_folder + os.sep + output_file_name
                shutil.copy(input_src_file, input_directory)
                shutil.copy(output_src_file, output_directory)

print("Done!")
