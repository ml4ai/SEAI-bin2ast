from typing import Union, List
import sys
import os
import json
import uuid
import platform
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
import timeit
import multiprocessing

import gen_c_prog as gen_c_prog
from bin2ast.gcc2cast.gcc_ast_to_cast import GCC2CAST
from batch_tokenize_instructions import extract_tokens_and_save  # TokenSet, extract_tokens_from_instr_file


# -----------------------------------------------------------------------------
# Multiprocessing-safe Progress Log
# -----------------------------------------------------------------------------

LOCK = multiprocessing.Lock()


PROGRESS_LOG_FILENAME = 'log_progress.txt'


def log_progress(log_path, instance_id):
    LOCK.acquire()
    with open(log_path, 'a') as file:
        file.write(f'{instance_id}\n')
    LOCK.release()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class Config:
    log_progress_root: str = ''
    corpus_root: str = ''
    num_samples: int = 1
    num_padding: int = 7
    total_attempts: int = 1
    base_name: str = ''

    gcc: str = ''
    gcc_plugin_filepath: str = ''
    cast_to_token_cast_filepath: str = ''
    ghidra_root: str = ''
    ghidra_script_root: str = ''
    ghidra_script_filename: str = ''

    stage_root_name: str = ''
    stage_root: str = ''
    src_root_name: str = ''
    src_root: str = ''
    cast_root_name: str = ''
    cast_root: str = ''
    bin_root_name: str = ''
    bin_root: str = ''
    ghidra_instructions_root_name: str = ''
    ghidra_instructions_root: str = ''
    corpus_input_tokens_root_name: str = ''   # tokenized binary
    corpus_input_tokens_root: str = ''
    corpus_output_tokens_root_name: str = ''  # tCAST
    corpus_output_tokens_root: str = ''

    time_start: float = 0
    iteration_times: List[float] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Load config
# -----------------------------------------------------------------------------

def missing_config_message():
    print("CONFIGURATION config.json")
    print("To use this script you must first create a file 'config.json'")
    print("with the following contents -- replace <path_to_ghidra_root> with the")
    print("appropriate absolute path within a string:")
    print("{")

    print("  \"log_progress_root\": \"<str> absolute path to the location the progress log will be stored\"")
    print("  \"corpus_root\": \"<str> absolute path to top-level root directory for the corpus\"")
    print("  \"num_samples\": \"<int> number of samples to generate\"")
    print("  \"num_padding\": \"<int> number of 0's to pad to the left of the filename sample index\"")
    print("  \"total_attempts\": \"<int> total sequential attempts to generate a valid, executable program\"")
    print("  \"base_name\": \"<str> the program base name\"")

    print("  \"gcc\": \"<str> path to gcc\"")
    print("  \"gcc_plugin_filepath\": \"<str> absolute path to gcc plugin\"")
    print("  \"cast_to_token_cast_filepath\": \"<str> absolute path to cast_to_token_cast.py script\"")
    print("  \"ghidra_root\": \"<str> root directory path of ghidra\"")
    print("  \"ghidra_script_root\": \"<str> root directory path of ghidra plugin scripts\"")
    print("  \"ghidra_script_filename\": \"<str> filename of ghidra plugin script\""
          "  # e.g., ..../DumpInstructionsByFunction.py")

    print("  \"stage_root\": \"<str> name of directory root for staging"
          " (generate candidate, attempt compile, attempt execution)\"")
    print("  \"src_root\": \"<str> name of directory root for C source code>\"")
    print("  \"cast_root\": \"<str> name of directory root for CAST>\"")
    print("  \"bin_root\": \"<str> name of directory root for binaries>\"")
    print("  \"ghidra_instructions_root\": \"<str> name of directory root for "
          "ghidra instructions (ghidra_working_directory)>\"")

    print("  \"corpus_input_tokens_dir_root\": \"<str> name of directory root for "
          "corpus input tokens (tokenized binaries)\"")
    print("  \"corpus_output_tokens_dir_root\": \"<str>: name of directory root for "
          "corpus output tokens (.tcast)\"")

    print("  \"sample_program_flag\": \"<Boolean>: whether to generate programs\"")
    print("  \"sample_program_num_samples\": \"<int>: number of programs to generate\"")
    print("  \"sample_program_base_name\": \"<str>: base name of all generated programs\"")
    print("}")


def configure_paths(config):
    config.stage_root = os.path.join(config.corpus_root, config.stage_root_name)
    config.src_root = os.path.join(config.corpus_root, config.src_root_name)
    config.cast_root = os.path.join(config.corpus_root, config.cast_root_name)
    config.bin_root = os.path.join(config.corpus_root, config.bin_root_name)
    config.ghidra_instructions_root = os.path.join(config.corpus_root, config.ghidra_instructions_root_name)
    config.corpus_input_tokens_root = os.path.join(config.corpus_root, config.corpus_input_tokens_root_name)
    config.corpus_output_tokens_root = os.path.join(config.corpus_root, config.corpus_output_tokens_root_name)


def load_config():
    # verify config.json exists
    if not os.path.isfile('config.json'):
        missing_config_message()
        raise Exception("ERROR: config.json not found; see CONFIGURATION message")

    config = Config()

    with open('config.json', 'r') as json_file:
        cdata = json.load(json_file)
        missing_fields = list()
        if 'log_progress_root' not in cdata:
            missing_fields.append('log_progress_root')
        else:
            config.log_progress_root = cdata['log_progress_root']
        if 'corpus_root' not in cdata:
            missing_fields.append('corpus_root')
        else:
            config.corpus_root = cdata['corpus_root']
        if 'num_samples' not in cdata:
            missing_fields.append('num_samples')
        else:
            config.num_samples = cdata['num_samples']
        if 'num_padding' not in cdata:
            missing_fields.append('num_padding')
        else:
            config.num_padding = cdata['num_padding']
        if 'total_attempts' not in cdata:
            missing_fields.append('total_attempts')
        else:
            config.total_attempts = cdata['total_attempts']
        if 'base_name' not in cdata:
            missing_fields.append('base_name')
        else:
            config.base_name = cdata['base_name']

        if 'gcc' not in cdata:
            missing_fields.append('gcc')
        else:
            config.gcc = cdata['gcc']
        if 'gcc_plugin_filepath' not in cdata:
            missing_fields.append('gcc_plugin_filepath')
        else:
            config.gcc_plugin_filepath = cdata['gcc_plugin_filepath']
        if 'cast_to_token_cast_filepath' not in cdata:
            missing_fields.append('cast_to_token_cast_filepath')
        else:
            config.cast_to_token_cast_filepath = cdata['cast_to_token_cast_filepath']
        if 'ghidra_root' not in cdata:
            missing_fields.append('ghidra_root')
        else:
            config.ghidra_root = cdata['ghidra_root']
        if 'ghidra_script_root' not in cdata:
            missing_fields.append('ghidra_script_root')
        else:
            config.ghidra_script_root = cdata['ghidra_script_root']
        if 'ghidra_script_filename' not in cdata:
            missing_fields.append('ghidra_script_filename')
        else:
            config.ghidra_script_filename = cdata['ghidra_script_filename']

        if 'stage_root_name' not in cdata:
            missing_fields.append('stage_root_name')
        else:
            config.stage_root_name = cdata['stage_root_name']
        if 'src_root_name' not in cdata:
            missing_fields.append('src_root_name')
        else:
            config.src_root_name = cdata['src_root_name']
        if 'cast_root_name' not in cdata:
            missing_fields.append('cast_root_name')
        else:
            config.cast_root_name = cdata['cast_root_name']
        if 'bin_root_name' not in cdata:
            missing_fields.append('bin_root_name')
        else:
            config.bin_root_name = cdata['bin_root_name']
        if 'ghidra_instructions_root_name' not in cdata:
            missing_fields.append('ghidra_instructions_root_name')
        else:
            config.ghidra_instructions_root_name = cdata['ghidra_instructions_root_name']

        if 'corpus_input_tokens_root_name' not in cdata:
            missing_fields.append('corpus_input_tokens_root_name')
        else:
            config.corpus_input_tokens_root_name = cdata['corpus_input_tokens_root_name']
        if 'corpus_output_tokens_root_name' not in cdata:
            missing_fields.append('corpus_output_tokens_root_name')
        else:
            config.corpus_output_tokens_root_name = cdata['corpus_output_tokens_root_name']

        if missing_fields:
            missing_config_message()
            missing_str = '\n  '.join(missing_fields)
            raise Exception("ERROR load_config(): config.json missing "
                            f"the following fields:\n  {missing_str}")

        return config


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def get_gcc_version(gcc_path):
    """
    Helper to extract the gcc version
    :param gcc_path: path to gcc
    :return: string representing gcc version
    """
    version_bytes = subprocess.check_output([gcc_path, '--version'])
    version_str = version_bytes.decode('utf-8')
    if 'clang' in version_str:
        clang_idx = version_str.find('clang')
        return version_str[clang_idx:version_str.find(')', clang_idx)], 'clang'
    else:
        return 'gcc-' + version_str.split('\n')[0].split(' ')[2], 'gcc'


def try_compile(config: Config, src_filepath: str, verbose: bool = False):
    """
    Attempts to compile binary from source using GCC with GCC-ast-dump plugin.
    :param config:
    :param src_filepath: C program source file path
    :param verbose:
    :return:
    """
    gcc_version, compiler_type = get_gcc_version(config.gcc)
    platform_name = platform.platform()
    binary_postfix = '__' + platform_name + '__' + gcc_version

    dst_filepath = os.path.splitext(src_filepath)[0] + binary_postfix

    # CTM 2022-05-10: Adding -lm to link the math lib
    command_list = [config.gcc, f'-fplugin={config.gcc_plugin_filepath}',
                    '-C', '-x', 'c++', '-O0', '-lm', src_filepath, '-o', dst_filepath]

    if verbose:
        print(f'try_compile() cwd {os.getcwd()}')
        print(f'              command_list: {command_list}')

    result = subprocess.run(command_list, stdout=subprocess.PIPE)

    if verbose:
        print(f'              result={result}')

    return result, dst_filepath


def gcc_ast_to_cast(filename_base: str, verbose_p: bool = False):
    ast_filename = filename_base + '_gcc_ast.json'
    ast_json = json.load(open(ast_filename))
    if verbose_p:
        print("gcc_ast_to_cast(): Translate GCC AST into CAST...")
    cast = GCC2CAST([ast_json]).to_cast()
    cast_filename = filename_base + '--CAST.json'
    json.dump(cast.to_json_object(), open(cast_filename, "w"))
    return ast_filename, cast_filename


def log_failure(filename_c:str, reason: str, archive_filename):
    with open('failures.log', 'a') as logfile:
        logfile.write(f'{filename_c}: {reason} : {archive_filename}\n')


def failure(location, _result, sample_prog_str, filename_src, filename_uuid_c):
    print(f'FAILURE - {location} - {_result}')
    print(f'CWD: {os.getcwd()}')
    print(f'listdir: {os.listdir()}')
    log_failure(filename_src, f'{location} return {_result}', filename_uuid_c)
    with open(filename_uuid_c, 'w') as out_file:
        out_file.write(sample_prog_str)
    # For some reason this won't work
    # -- always results in "No such file or directory"
    # subprocess.call(['cp ' + filename_src + ' ' + filename_uuid_c])


def finalize(config: Config):  # , token_set: TokenSet):  ## TODO CTM 2022-05-10: update once new version of TokenSet is implemented
    token_set_summary_filepath = os.path.join(config.stage_root, 'tokens_summary.txt')
    original_stdout = sys.stdout
    time_end = timeit.default_timer()
    total_time = time_end - config.time_start
    with open(token_set_summary_filepath, 'w') as fout:
        sys.stdout = fout
        # token_set.print()
        print(f'finalize(): total time: {config.time_start} {time_end} {total_time}')
        print(f'            iteration_times: {config.iteration_times}')
        sys.stdout = original_stdout


def try_generate(config: Config, i: int,  # token_set: TokenSet,  ## TODO CTM 2022-05-10: update once new version of TokenSet is implemented
                 verbose=False):

    if verbose:
        print('try_generate(): i={i}')

    log_progress_path = os.path.join(config.log_progress_root, PROGRESS_LOG_FILENAME)

    time_start = timeit.default_timer()

    num_str = f'{i}'.zfill(config.num_padding)
    filename_base = f'{config.base_name}_{num_str}'
    filename_src = filename_base + '.c'

    filename_src_stats = filename_base + '_stats.txt'

    filename_bin = ''
    filename_gcc_ast = ''
    filename_cast = ''
    filename_ghidra_instructions = ''
    filename_tokens_output = ''
    sample_prog: Union[gen_c_prog.ProgramSpec, None] = None

    attempt = 0
    keep_going = True
    success = False

    ghidra_command = os.path.join(config.ghidra_root, 'support/analyzeHeadless')
    ghidra_project_name = 'temp_ghidra_project'
    ghidra_project_dir = '.'

    while keep_going:

        # stop if have exceeded total_attempts
        if attempt >= config.total_attempts:
            break

        attempt += 1

        temp_uuid = str(uuid.uuid4())
        filename_base_uuid = f'{filename_base}_{temp_uuid}'
        filename_uuid_c = filename_base_uuid + '.c'

        if verbose:
            print(f'    attempt={attempt} : {filename_uuid_c}')

        # generate candidate source code
        sample_prog, sample_program_str = \
            gen_c_prog.generate_and_save_program(filename_src)

        # save sample stats
        # TODO CTM 2022-05-10: implement new stats collection and call here...
        #  When re-introduce, need to update function mv_files, below
        # with open(filename_src_stats, 'w') as stats_file:
        #     stats_file.write(gen_c_prog.ExprSeqSampleStats(sample_prog).to_string())

        # compile candidate
        result, filename_bin = try_compile(config=config, src_filepath=filename_src, verbose=verbose)  # filepath_uuid_c)

        if result.returncode != 0:
            failure('COMPILE', result, sample_program_str, filename_src, filename_uuid_c)
            # print(f'FAILURE - COMPILE - {result.returncode}')
            # print(f'CWD: {os.getcwd()}')
            # print(f'listdir: {os.listdir()}')
            # log_failure(filename_src, f'compilation return {result}')
            # subprocess.call(['cp ' + filename_src + ' ' + filename_uuid_c])
            continue

        # CTM 2022-05-10: For now, skipping execution test
        # # execute_candidate
        # result = subprocess.run([f'./{filename_bin}'], stdout=subprocess.PIPE)
        #
        # if result.returncode != 0:
        #     failure('EXECUTE', result, sample_program_str, filename_src, filename_uuid_c)
        #     # print(f'FAILURE - EXECUTE - {result.returncode}')
        #     # print(f'CWD: {os.getcwd()}')
        #     # print(f'listdir: {os.listdir()}')
        #     # log_failure(filename_src, f'execution return {result.returncode}')
        #     # subprocess.call(['cp ' + filename_src + ' ' + filename_uuid_c])
        #     continue

        # gcc ast to CAST
        filename_gcc_ast, filename_cast = gcc_ast_to_cast(filename_base, verbose_p=verbose)

        # run Ghidra
        filename_ghidra_instructions = filename_bin + '-instructions.txt'
        command_list = \
            [ghidra_command, ghidra_project_dir, ghidra_project_name,
             '-import', filename_bin,
             '-scriptPath', config.ghidra_script_root,
             '-postScript', config.ghidra_script_filename,
             '-deleteProject']
        result = subprocess.run(command_list, stdout=subprocess.PIPE)

        if result.returncode != 0:
            failure('GHIDRA', result, sample_program_str, filename_src, filename_uuid_c)
            # print(f'CWD: {os.getcwd()}')
            # print(f'listdir: {os.listdir()}')
            # print(f'FAILURE - GHIDRA - {result.returncode}')
            # log_failure(filename_src, f'ghidra return {result.returncode}')
            # subprocess.call(['cp ' + filename_src + ' ' + filename_uuid_c])
            continue

        # tokenize CAST
        filename_tokens_output = filename_base + '--CAST.tcast'
        result = subprocess.run(['python', config.cast_to_token_cast_filepath, '-f',
                                 filename_cast], stdout=subprocess.PIPE)

        if result.returncode != 0:
            failure('TOKEN_CAST', result, sample_program_str, filename_src, filename_uuid_c)
            # print(f'CWD: {os.getcwd()}')
            # print(f'listdir: {os.listdir()}')
            # print(f'FAILURE - cast_to_token_cast.py - {result.returncode}')
            # log_failure(filename_src, f'cast_to_token_cast return {result.returncode}')
            # subprocess.call(['cp ' + filename_src + ' ' + filename_uuid_c])
            continue

        # if get this far, then success!
        success = True
        keep_going = False  # could also just break...

    if success:
        # extract bin input tokens from ghidra instructions
        filename_bin_tokens = f'{config.corpus_input_tokens_root}/{filename_bin}__tokens.txt'

        extract_tokens_and_save(_src_filepath=filename_ghidra_instructions,
                                _dst_filepath=filename_bin_tokens)

        # CTM 2022-05-10: the old method:
        # extract_tokens_from_instr_file(token_set=token_set,
        #                                _src_filepath=filename_ghidra_instructions,
        #                                _dst_filepath=filename_bin_tokens,
        #                                execute_p=True,
        #                                verbose_p=False)

        # move files to respective locations:
        mv_files(config,  # token_set,  ## TODO CTM 2022-05-10: update once new version of TokenSet is implemented
                 filename_src,  filename_src_stats,
                 filename_bin, filename_gcc_ast, filename_cast,
                 filename_ghidra_instructions,
                 filename_tokens_output)

        # Update the counter file
        with open('counter.txt', 'w') as counter_file:
            counter_file.write(f'{i}, {config.num_padding}')

        # Log progress!
        log_progress(log_progress_path, num_str)

        print('Success')

        time_end = timeit.default_timer()
        time_elapsed = time_end - time_start
        with open('iteration_times.txt', 'a') as it_file:
            it_file.write(f'{time_elapsed} ')
        config.iteration_times.append(time_elapsed)
    else:
        time_end = timeit.default_timer()
        time_elapsed = time_end - time_start
        with open('iteration_times.txt', 'a') as it_file:
            it_file.write(f'{time_elapsed} ')
        config.iteration_times.append(time_elapsed)

        # save current token_set before bailing
        finalize(config)  # TODO CTM 2022-05-10: update once new version of TokenSet is implemented
        # finalize(config, token_set)

        raise Exception(f"ERROR try_generate(): failed to generate a viable program after {attempt} tries.")


def mv_files(config,  # token_set, ## TODO CTM 2022-05-10: update once new version of TokenSet is implemented
             filename_src,  filename_src_stats,
             filename_bin, filename_gcc_ast, filename_cast,
             filename_ghidra_instructions,
             filename_tokens_output):
    filenames = (filename_src,  # filename_src_stats,  ## TODO CTM 2022-05-10: turning off mv of _stats file for now
                 filename_bin, filename_gcc_ast, filename_cast,
                 filename_ghidra_instructions,
                 filename_tokens_output)
    dest_paths = (config.src_root,  # config.src_root,  ## TODO CTM 2022-05-10: turning off mv of _stats file for now
                  config.bin_root, config.cast_root, config.cast_root,
                  config.ghidra_instructions_root,
                  config.corpus_output_tokens_root)
    for src_filename, dest_path in zip(filenames, dest_paths):
        dest_filepath = os.path.join(dest_path, src_filename)
        result = subprocess.run(['mv', src_filename, dest_filepath], stdout=subprocess.PIPE)
        if result.returncode != 0:
            finalize(config)  # TODO CTM 2022-05-10: update once new version of TokenSet is implemented
            # finalize(config, token_set)
            raise Exception(f"ERROR mv_files(): failed mv {src_filename} --> {dest_filepath}\n"
                            f"  current directory: {os.getcwd()}")


def generate_corpus(start=0, num_samples=None, corpus_root=None, verbose=False):
    config = load_config()

    config.time_start = timeit.default_timer()

    if corpus_root:
        print(f'NOTE: Overriding config.corpus_root with {corpus_root}')
        config.corpus_root = corpus_root

    if num_samples:
        print(f'NOTE: Overriding config.num_samples with {num_samples}')
        config.num_samples = num_samples

    configure_paths(config)

    print(f'DEBUG: config.corpus_root: {config.corpus_root}')
    print(f'DEBUG: config.num_smaples: {config.num_samples}')

    # Create corpus root, but don't allow if directory already exists,
    # to prevent overwriting...
    for path in (config.corpus_root, config.stage_root, config.src_root,
                 config.cast_root, config.bin_root,
                 config.ghidra_instructions_root,
                 config.corpus_input_tokens_root,
                 config.corpus_output_tokens_root):
        Path(path).mkdir(parents=True, exist_ok=False)

    # token_set = TokenSet()  ## TODO CTM 2022-05-10: update once new version of TokenSet is implemented

    original_working_dir = os.getcwd()
    os.chdir(config.stage_root)

    print(f'CWD: {os.getcwd()}')

    # verify that we can find cast_to_token_cast.py script
    if os.path.isfile(config.cast_to_token_cast_filepath):
        print(f"NOTE: Found cast_to_token_cast.py: {config.cast_to_token_cast_filepath}")
    else:
        raise Exception(f"ERROR: Cannot find cast_to_token_cast.py: {config.cast_to_token_cast_filepath}")

    for i in range(start, start + config.num_samples):
        try_generate(config=config, i=i,  # token_set=token_set, ## TODO CTM 2022-05-10: update once new version of TokenSet is implemented
                     verbose=verbose)
    finalize(config)  ## TODO CTM 2022-05-10: update once new version of TokenSet is implemented
    # finalize(config, token_set)
    os.chdir(original_working_dir)


# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_corpus()
