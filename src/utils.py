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

def list_join(lst, join_with):
    joined = []
    for i, item in enumerate(lst):
        joined += item
        if not i==len(lst)-1:
            joined += [join_with]
    return joined


def drop_from_iterable(lst, drop_elements):
    if isinstance(drop_elements, type(lst[0])):
        drop_elements = set(list(drop_elements))
    elif isinstance(drop_elements, list):
        drop_elements = set(drop_elements)
    
    return [x for x in lst if not x in drop_elements]


def load_data_file(dir_or_path):
    if dir_or_path.endswith('.json'):
        return load_json(dir_or_path)
    else:
        return load_json(dir_or_path + '2^%d.json'%args['data']['max_pow'])

def get_best_checkpoint(path, map_location=None):
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
    return torch.load(chosen_path, map_location=map_location)


def get_last_checkpoint(path, map_location):
    if not 'checkpoints' in path:
        path = os.path.join(path, 'checkpoints/')
    files = [f for f in os.listdir(path) if f.endswith('.pt')]
    steps = [int(fname.split('_')[0]) for fname in files]
    latest_model_idx = np.argmax(steps)
    chosen_path = path + files[latest_model_idx]
    print('Loading model at %s'%chosen_path)
    return torch.load(chosen_path, map_location=map_location)

    
def update_args_with_cli(args, input_args):
    args['metrics']['max_num'] = input_args.max_num
    args['metrics']['save_suffix'] = input_args.suffix
    if input_args.n_beams > 0:
        args['metrics']['n_beams'] = input_args.n_beams
    
    if 'checkpoints' in input_args.path:
        path = input_args.path
        args['io']['save_path'] = path[:path.find('checkpoints')]
    else:
        args['io']['save_path'] = input_args.path
    if input_args.data_loc:
        args['data']['test_path'] = input_args.data_loc
    if input_args.temperature > 0:
        args['metrics']['temperature'] = input_args.temperature
    args['loader']['test']['num_workers'] = input_args.num_workers
    return args

def backfill_args(args):
    # Add default arguments for things that werne't features when the model was run
    
    return args



"""Miller Rabin test from same source as Quadratic-Seive in scripts"""
def _try_composite(a, d, n, s):
    if pow(a, d, n) == 1:
        return False
    for i in range(s):
        if pow(a, 2**i * d, n) == n-1:
            return False
    return True # n  is definitely composite



def is_prime(n, _precision_for_huge_n=16):
    n = int(n)
    if n < 2: return False
    if n <= _known_primes[-1]:
        if n in _known_primes:
            return True
    for p in _known_primes:
        if n % p == 0:
            return False
    d, s = n - 1, 0
    while not d % 2:
        d, s = d >> 1, s + 1
    # Returns exact according to http://primes.utm.edu/prove/prove2_3.html
    if n < 1373653: 
        return not any(_try_composite(a, d, n, s) for a in (2, 3))
    if n < 25326001: 
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5))
    if n < 118670087467: 
        if n == 3215031751: 
            return False
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7))
    if n < 2152302898747:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11))
    if n < 3474749660383: 
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13))
    if n < 341550071728321:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13, 17))
    if n < 3_825_123_056_546_413_051:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13, 17, 19, 23))
    if n < 18_446_744_073_709_551_616:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37))
    return not any(_try_composite(a, d, n, s) 
                for a in _known_primes[:_precision_for_huge_n])

_known_primes = [2, 3]
_known_primes += [x for x in range(5, 1000, 2) if is_prime(x)]