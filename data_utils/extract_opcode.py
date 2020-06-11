from tqdm import tqdm
import multiprocessing
from config import *
from subprocess import Popen, PIPE
import random
import json
from data_utils.data_loaders import *


def generate_opcode(bin_filename, text_filename, debug=False):
    list_of_cmd_args = [
        ['objdump', '-j .text', '-D', bin_filename],
        ['objdump', '-j CODE', '-D', bin_filename],
        ['objdump', '-d', bin_filename]
    ]

    got_upcode = False
    asm_code = ''

    for cmd_num, cmd_args in enumerate(list_of_cmd_args):
        try:
            if debug:
                print(f'cmd_num = {cmd_num}')
                print(f'cmd_args = {" ".join(cmd_args)}')
            process = Popen(cmd_args, stdout=PIPE, stderr=PIPE)
            p_out, p_err = process.communicate()
            if debug:
                print(p_out)
            asm_code = str(p_out).split('\\n')
        except ValueError:
            got_upcode = False
        else:
            if len(asm_code) > 5:
                got_upcode = True

        if got_upcode:
            break

    if got_upcode:
        with open(text_filename, 'w+') as f:
            for line in asm_code:
                line = line.split('\\t')
                if len(line) > 2:
                    opcode_line = line[2]
                    opcode_line = opcode_line.split(' ')
                    if len(opcode_line) > 0:
                        f.write(opcode_line[0])
                        f.write('\n')

    else:
        # TODO some files are empty. check generate_opcode
        if debug:
            print(f'Giving up on {bin_filename}')


def process_opcodes_bulk(input_dir, output_dir=ORG_DATASET_OPCODES_PATH, max_files=0):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    list_dirs = os.listdir(input_dir)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:

        jobs = []
        results = []
        total_count = 0

        for dirname in list_dirs:
            list_files = os.listdir(os.path.join(input_dir, dirname))
            count = 0
            for filename in list_files:
                input_filename = os.path.join(input_dir, dirname, filename)
                try:
                    output_filename = os.path.splitext(os.path.basename(input_filename))[0] + '.txt'
                    output_class_dir = os.path.join(output_dir, dirname)
                    if not os.path.isdir(output_class_dir):
                        os.mkdir(output_class_dir)
                    output_filename = os.path.join(output_dir, dirname, output_filename)

                    jobs.append(
                        pool.apply_async(generate_opcode, (input_filename, output_filename)))
                    count += 1
                    if max_files > 0 and max_files == count:
                        break
                except:
                    print('Ignoring ', filename)

            total_count += count
        tqdm_desc = 'Extracting opcodes from Malware bins'
        for job in tqdm(jobs, desc=tqdm_desc):
            results.append(job.get())


def get_jason_filename(combined_path_for_json, key, class_name):
    return os.path.join(combined_path_for_json, class_name, key + '_' + class_name + '.json')


def split_opcodes(input_class_dir, opcode_len=-1, train_split=0.8):
    """
    input_class_dir contains opcode json file.
    Create train and test split for this json file
    """
    opcode_path = get_opcode_datapath(opcode_len, check_exist=False)
    combined_path_for_json = opcode_path['combined_path_for_json']

    list_files = os.listdir(input_class_dir)
    class_name = os.path.basename(input_class_dir)
    random.shuffle(list_files)
    total_samples = len(list_files)
    train_size = int(total_samples * train_split)
    test_size = total_samples - train_size

    train_files = list_files[0:train_size]
    test_files = list_files[train_size:]

    samples = {'train': train_files, 'test': test_files}

    for key in samples.keys():
        sample_json = []
        all_filenames = samples[key]
        for filename in all_filenames:
            input_filename = os.path.join(input_class_dir, filename)
            opcodes = []
            with open(input_filename, 'r') as f:
                opcodes = f.read().splitlines()

            # print(opcodes)
            if opcode_len != -1:
                opcodes = opcodes[0:opcode_len]
            if len(opcodes) < 1:
                continue

            dir_name = os.path.join(ORG_DATASET_ROOT_PATH, combined_path_for_json)
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)
            out_json_file = get_jason_filename(dir_name, key, class_name)
            with open(out_json_file, 'a') as outfile:
                file_dict = {'text': opcodes, 'label': class_name}
                json.dump(file_dict, outfile)
                outfile.write('\n')


def process_split_opcodes(root_datapath, opcode_len=-1):
    opcode_path = get_opcode_datapath(opcode_len, check_exist=False)
    train_split_json = os.path.join(ORG_DATASET_ROOT_PATH, opcode_path['train_split_json'])
    test_split_json = os.path.join(ORG_DATASET_ROOT_PATH, opcode_path['test_split_json'])
    combined_path_for_json = opcode_path['combined_path_for_json']

    classes = os.listdir(root_datapath)
    #
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for class_name in classes:
            class_dir = os.path.join(root_datapath, class_name)
            jobs.append(pool.apply_async(split_opcodes, (class_dir, opcode_len,)))

        for job in tqdm(jobs, desc="Generating opcodes with json and splitting for opcode len = {opcode_len}".format(
                opcode_len=opcode_len)):
            results.append(job.get())

    print(f'Merging all training and testing classes')
    samples = {'train': train_split_json,
               'test': test_split_json}

    for key in samples.keys():
        out_json_file = samples[key]
        with open(out_json_file, 'a') as outfile:
            for class_name in classes:
                json_file = os.path.join(ORG_DATASET_ROOT_PATH,
                                         get_jason_filename(combined_path_for_json, key, class_name))
                print(f'loading {json_file}')
                count = 0
                with open(json_file, 'r') as fp:
                    while True:
                        line = fp.readline()
                        if not line:
                            break
                        count += 1
                        try:
                            # write only right formated lines
                            json.loads(line)
                        except:
                            print(f'Error in {json_file} at count {count}')
                        else:
                            outfile.write(line)

    print(f'Merged all training and testing classes {samples}')
