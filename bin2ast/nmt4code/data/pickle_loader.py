"""
load pickled data instead of raw txt files to make the data loading faster
"""
import pickle


def load_pickled_data(pickled_data_path):
    # print("loading data...")
    # load pickled files and print some stats
    with open(pickled_data_path + "/" + "stats.pickle", 'rb') as read_file:
        stats = pickle.load(read_file)

    # load input lang: for train data augmentation
    with open(pickled_data_path + "/" + "input_lang.pickle", 'rb') as read_file:
        input_lang = pickle.load(read_file)

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

    pickled_data = {'stats': stats,
                    'input_lang': input_lang,
                    'output_lang': output_lang,
                    'train_input_data': train_input_data,
                    'train_target_data': train_target_data,
                    'val_input_data': val_input_data,
                    'val_target_data': val_target_data,
                    'test_input_data': test_input_data,
                    'test_target_data': test_target_data}

    return pickled_data
