"""
general util functions
"""

from torch import nn
import ast
from collections import OrderedDict
from typing import Dict


def init_weights(m):
    """
    initialize the weights of module
    :param m: module
    :return: none
    """
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def epoch_time(s_time, e_time):
    """
    time taken to train one epoch
    :param s_time: start time
    :param e_time: end time
    :return: time elapsed
    """
    elapsed_time = e_time - s_time
    elapsed_mins = int(elapsed_time / 60.0)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60.0))
    return elapsed_mins, elapsed_secs


def init_weights_attn(m):
    """
    initialize the weights for the modules of encoder and decoder with attention model
    :param m: module
    :return: None
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def init_weights_transformer(m):
    """
    initialize the weights for the modules of encoder and decoder with transformer model
    :param m: module
    :return: None
    """
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def init_weights_tree_decoder(m):
    """
    initialize weights for the modules of encoder and decoder with transformer encoder tree decoder
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.1, 0.1)
        else:
            nn.init.constant_(param.data, 0)


def init_weights_gnn_decoder(m):
    """
    initialize weights for the modules of encoder and decoder with transformer encoder tree decoder
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.1, 0.1)
        else:
            nn.init.constant_(param.data, 0)


def hexadecimaltodecimal(value):
    """
    convert hexadecimal to binary (2's complement) -> decimal
    :param value: hex value
    :return: decimal value
    """
    # translate hex value -> binary (2's complement) -> decimal
    # need a dictionary that maps every float number into 4 bit binary number
    hex_to_bin = {"0": "0000", "1": "0001", "2": "0010", "3": "0011", "4": "0100",
                  "5": "0101", "6": "0110", "7": "0111", "8": "1000", "9": "1001",
                  "a": "1010", "b": "1011", "c": "1100", "d": "1101", "e": "1110",
                  "f": "1111"}
    result = ""
    # remove the initial 0x if it exists
    if value.startswith("-0x"):
        value = value[3:]
        result = -(int(value, 16))
        return str(result)
    elif value.startswith("0x"):
        value = value[2:]
    # convert to lower case
    value = value.lower()
    # for each digit convert it to binary
    for item in value:
        try:
            result += hex_to_bin[item]
        except KeyError:
            print("Unrecognized characters in the hexadecimal conversion")
            return
    # if the initial value is zero: it's a positive number
    if result[0] == "0":
        # convert it to integer
        result = int(result, 2)
    # else the number is negative, convert it differently
    else:
        # flip the bits
        temp = ''.join(['1' if i == '0' else '0' for i in result])
        # add 1 and convert to decimal => convert to decimal and add 1
        result = -(int(temp, 2) + 1)
    return str(result)


def get_input_tokens_list_v2_old(input_path, with_values=True, fix_hex_values=True):
    """
    This is for v2 dataset: need to develop similar for v3 dataset
    given the input_path for tokens.txt file, it will parse the input txt file and find out
    values for _v0, _v1 and so on
    :param input_path: path to the input file
    :param with_values: if with_values set to on, _v0 will map to actual values like v_start, 1, 2, v_end
    else it will be set to original _v0
    :param fix_hex_values: if we want to fix the hex to deciaml conversion using 2's complement
    :return: list of tokens
    """
    # boolean to determine if we need to parse the given line
    parse = False
    with open(input_path, 'r') as read_file:
        tokens = read_file.readline().strip()
        values_map = {}
        for line in read_file:
            line = line.strip()
            if line.startswith("START token_map_val"):
                parse = True
                continue
            if line.startswith("END token_map_val"):
                break
            if parse:
                # parse the values line to get the exact values
                if "Answer:" in line:
                    # will parse the final printf line with Answer
                    value_token, value, _, metadata, _ = line.split(":")
                else:
                    try:
                        # will parse most of the remaining lines
                        value_token, value, _, metadata = line.split(":")
                    except ValueError:
                        # will parse the '_v8 : [RAX + RDX*0x1] : 0x1' format
                        try:
                            value_token, registers, value = line.split(":")
                            metadata = "interpreted_hex"
                        except ValueError:
                            # will try to parse the following format: _v14 : qword ptr [0x00402018]
                            # where I don't know what to extract
                            # so I will extract the value in brackets for now
                            value_token, value = line.split(":")
                            start = value.index("[")
                            end = value.index("]")
                            value = value[start + 1: end]
                            metadata = "interpreted_hex"

                value_token = value_token.strip()
                value = value.strip()
                if metadata.startswith("interpreted_hex_float"):
                    _, _, actual_value, _ = metadata.split(",")
                    actual_value = str(actual_value.strip())
                elif metadata.startswith("interpreted_hex"):
                    if fix_hex_values:
                        actual_value = hexadecimaltodecimal(value)
                    else:
                        # extract hex value from metadata part
                        _, _, actual_value, _ = metadata.split(",")
                        actual_value = str(actual_value.strip())
                elif metadata.startswith("string"):
                    actual_value = "Answer"
                else:
                    print(f"unknown format in file: {input_path}")
                    return

                if actual_value == "Answer":
                    values_map[value_token] = actual_value
                else:
                    result = ["v_start"]
                    for item in actual_value:
                        result.append(item)
                    result.append("v_end")
                    values_map[value_token] = result

        # modify tokens with values_map dict
        if with_values:
            new_tokens = tokens.replace("'", "")
            new_tokens = new_tokens.split(",")
            tokens_list = []
            for token in new_tokens:
                token = token.strip()
                if token in values_map:
                    val = values_map[token]
                    if val == "Answer":
                        tokens_list.append(val)
                    else:
                        tokens_list.extend(values_map[token])
                else:
                    tokens_list.append(token)
        else:
            tokens = tokens.replace("'", "")
            tokens_list = tokens.split(",")

        return tokens_list


def get_input_tokens_list_v2(input_path, fix_hex_values=True):
    """
    This is for v2 dataset: It will return the token sequence and val_dict
    val_dict contains _v# as the keys and the actual values [6.2] as the values
    use get_input_tokens_list_v2_old: for experiment with values and no values
    """
    val_dict = dict()
    # boolean to determine if we need to parse the given line
    parse = False
    with open(input_path, 'r') as read_file:
        tokens = read_file.readline().strip()
        for line in read_file:
            line = line.strip()
            if line.startswith("START token_map_val"):
                parse = True
                continue
            if line.startswith("END token_map_val"):
                break
            if parse:
                # parse the values line to get the exact values
                if "Answer:" in line:
                    # will parse the final printf line with Answer
                    value_token, value, _, metadata, _ = line.split(":")
                else:
                    try:
                        # will parse most of the remaining lines
                        value_token, value, _, metadata = line.split(":")
                    except ValueError:
                        continue

                value_token = value_token.strip()
                value = value.strip()
                if metadata.startswith("interpreted_hex_float"):
                    _, _, actual_value, _ = metadata.split(",")
                    actual_value = str(round(float(actual_value.strip()), 1))
                elif metadata.startswith("interpreted_hex"):
                    if fix_hex_values:
                        actual_value = str(round(float(hexadecimaltodecimal(value)), 1))
                    else:
                        # extract hex value from metadata part
                        _, _, actual_value, _ = metadata.split(",")
                        actual_value = str(round(float(actual_value.strip()), 1))
                elif metadata.startswith("string"):
                    actual_value = "Answer"
                else:
                    print(f"unknown format in file: {input_path}")
                    return

                val_dict[value_token] = actual_value

        # modify tokens with values_map dict
        tokens = tokens.replace("'", "")
        tokens_list = tokens.split(",")

    # tokens_list has tokens with space for some reason, fix them
    new_token_list = []
    for token in tokens_list:
        new_token_list.append(token.strip())
    return new_token_list, val_dict


def get_output_tokens_list_v2(output_path):
    """
    This is for v2 dataset: returns output token list and a dictionary of values
    for with_values and no_values experiment: use get_output_tokens_list_v2_old function
    """
    val_dict = dict()
    parse = False
    with open(output_path, 'r') as read_file:
        tokens = read_file.readline().strip()
        counter = 0
        for line in read_file:
            if "VALUES" in line:
                parse = True
                continue
            elif parse and line == "\n":
                break
            elif parse:
                value = str(line.strip())
                if "Answer" in value:
                    result = "Answer"
                else:
                    result = str(round(float(value), 1))
                val_dict["Val" + str(counter)] = result
                counter += 1
    tokens_list = tokens.split()
    return tokens_list, val_dict


class Tokens:
    """
    class that represents key-value pair for different tokens (value, param, address)
    """

    def __init__(self, name: str, base: str):
        """
        name of token: value, param, address
        counter: start value for each token
        token_to_elm: ordered dict for token to element
        """
        self.name = name
        self.base = base
        self.counter = 0
        # key value pair from key to value
        self.token_to_elm: Dict[str, str] = OrderedDict()
        # reverse dictionary of self.elm_to_token: for efficient search to find if elm exists already
        self.elm_to_token: Dict[str, str] = dict()

    def add_token(self, elm: str) -> str:
        """
        elm: element to add to the token dict
        return: the key to which the elm was assigned to or the previous key if
        the element already exists: same id for repeating tokens
        """
        # if the elm already exists in the dict, just return the key of that element
        if elm in self.elm_to_token:
            return self.elm_to_token[elm]
        # if not assign the elm to new key and return new key
        key = self.base + str(self.counter)
        # update key-elm dict
        self.token_to_elm[key] = elm
        # update elm-key dict
        self.elm_to_token[elm] = key
        self.counter += 1
        return key


def get_input_tokens_list_v3(input_path):
    """
    given the input_path for tokens.txt file, it will parse the input txt file and return
    a dictionary: which contains function_name: list of tokens for that function
    :return: dictionary function_name: list of tokens
    Each function has a local view: this one creates tokens per function
    may be we need something different to reconstruct the whole program
    :returns the dictionary for each function val_dict[func_name] = dict()
    dict() has keys as _v# and values (actual values)
    also return a dictionary for globals
    """
    token_dict = dict()
    val_dict = dict()
    global_dict = dict()
    with open(input_path, 'r') as read_file:
        inside_globals = False
        func_name = None
        inside_value_token_map = False
        for line in read_file:
            if "global_tokens_map" in line:
                inside_globals = True

            elif inside_globals and line == "\n":
                inside_globals = False

            elif inside_globals:
                global_token, global_tuple = line.strip().split(":", 1)
                try:
                    _, _, decimal_value = global_tuple.split(",")
                except ValueError:
                    continue
                global_dict[global_token] = str(round(float(decimal_value.strip().replace(")", "")), 1))

            elif "function_name" in line:
                func_name = line.split(":")[-1].strip()
                next(read_file)  # skip one line: token sequence line
                token_seq = ast.literal_eval(next(read_file).strip())
                # rename called function to be sequential: call _f0, call _f1 etc
                tokens = Tokens(name="function", base="_f")
                new_seq = []
                for item in token_seq:
                    if "_f" in item:
                        key = tokens.add_token(item)
                        new_seq.append(key)
                    else:
                        new_seq.append(item)
                token_dict[func_name] = new_seq

            elif "value_tokens_map" in line:
                inside_value_token_map = True
                val_dict[func_name] = dict()

            elif inside_value_token_map and line == "\n":
                inside_value_token_map = False

            elif inside_value_token_map:
                value_token, value_tuple = line.strip().split(":", 1)
                try:
                    _, _, value_decimal = ast.literal_eval(value_tuple)
                except:
                    continue
                value_decimal = str(round(float(value_decimal), 1))
                val_dict[func_name][value_token] = value_decimal

    return token_dict, val_dict, global_dict


def get_output_tokens_list_v3(output_path):
    """
    given the output_path for --CAST.tcast file, it will parse that file and return
    a dictionary: which contains function_name: list of tokens for that function
    :return: dictionary function_name: list of tokens
    Each function has a local view: this one creates tokens per function
    may be we need something different to reconstruct the whole program
    return_val_dict: returns the dictionary for each function val_dict[func_name] = dict()
    dict() has keys as _v# and values (actual values)
    """
    token_dict = dict()
    val_dict = dict()
    global_dict = dict()
    with open(output_path, 'r') as read_file:
        inside_globals = False
        func_name = None
        inside_value_token_map = False
        for line in read_file:
            if "global_tokens_map" in line:
                inside_globals = True

            elif inside_globals and line == "\n":
                inside_globals = False

            elif inside_globals:
                global_token, decimal_value = line.strip().split(":")
                try:
                    decimal_value = str(round(float(decimal_value), 1))
                except ValueError:
                    decimal_value = decimal_value.split("=")[-1].strip()
                    decimal_value = str(round(float(decimal_value), 1))

                global_dict[global_token] = decimal_value

            if "function_name" in line:
                func_name = line.split(":")[-1].strip()
                next(read_file)  # skip one line: token sequence line
                token_seq = next(read_file).strip()
                token_seq = token_seq.split(' ')
                # replace function name (main, fn0, fn1, fn2, etc with common token)
                token_seq[1] = "name"
                # rename called function to be sequential: call _f0, call _f1 etc
                tokens = Tokens(name="function", base="_f")
                new_seq = []
                for item in token_seq:
                    if "_f" in item:
                        key = tokens.add_token(item)
                        new_seq.append(key)
                    else:
                        new_seq.append(item)
                token_dict[func_name] = new_seq

            elif "value_tokens_map" in line:
                inside_value_token_map = True
                val_dict[func_name] = dict()

            elif inside_value_token_map and line == "\n":
                inside_value_token_map = False

            elif inside_value_token_map:
                value_token, value_decimal = line.strip().split(":", 1)
                value_decimal = str(round(float(value_decimal), 1))
                val_dict[func_name][value_token] = value_decimal

    return token_dict, val_dict, global_dict


def is_exact_match_possible(input_dict, output_dict):
    """
    check if exact match is possible for copy mechanism
    return: true or false
    """
    input_values = list(input_dict.values())
    for key, value in output_dict.items():
        if value not in input_values:
            return False
    return True
