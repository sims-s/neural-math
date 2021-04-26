import json
import os
import torch
import numpy as np

def save_json(to_save, path):
    with open(path, 'w') as f:
        json.dump(to_save, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_max_input_size(max_power, input_padding):
    input_size = max_power + (2 if input_padding=='pad' else 0)
    return input_size

def load_data_file(dir_or_path):
    if dir_or_path.endswith('.json'):
        return load_json(dir_or_path)
    else:
        return load_json(dir_or_path + '2^%d.json'%args['data']['max_pow'])

def get_max_decode_size(max_power):
    return 3*max_power + 1

def get_best_checkpoint(path):
    if path.endswith('.pt'):
        chosen_path = path
    else:
        if not 'checkpoints' in path:
            path = os.path.join(path, 'checkpoints/')
        files = [f for f in os.listdir(path) if f.endswith('.pt')]
        losses = [float(fname.split('_')[1][:-3]) for fname in files]
        best_loss_idx = np.argmin(losses)
        chosen_path = path + files[best_loss_idx]
    print('Loading model at %s'%chosen_path)
    return torch.load(chosen_path)
    
def update_args_with_cli(args, input_args):
    args['metrics']['max_num'] = input_args.max_num
    args['metrics']['save_suffix'] = input_args.suffix
    if input_args.n_beams > 0:
        args['metrics']['n_beams'] = input_args.n_beams
    args['io']['save_path'] = input_args.path
    if input_args.data_loc:
        args['data']['data_loc'] = input_args.data_loc
    return args

def backfill_args(args):
    # Add default arguments for things that werne't features when the model was run
    return args