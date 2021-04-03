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

def get_max_decode_size(max_power):
    return 3*max_power

def get_target_checkpoint(path):
    if path.endswith('.pt'):
        chosen_path = path
    else:
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
    return args

def backfill_args(args):
    # Add default arguments for things that werne't features when the model was run
    if not 'max_grad_norm' in args['optimizer']:
        args['max_grad_norm'] = -1
    return args