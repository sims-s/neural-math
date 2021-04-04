from argparse import ArgumentParser
from tqdm.auto import tqdm
import os
import sys
import numpy as np
sys.path.append('./src/')
from utils import save_json
sys.path.append('./src/PrimeFactorization/')
from PrimeFactorization import primeFactorize


def get_bounds(args):
    ub = args.max_value if args.max_power < 0 else 2**args.max_power
    if ub < 0:
        raise ValueError('Need max_val or max_pow as input!')
    lb = args.min_value if args.min_power < 0 else 2**args.min_power
    return lb, ub

def make_data(lb, ub):
    data = {}
    pbar = tqdm(total = ub-lb, leave=False)
    for i in range(lb, ub):
        data[i] = primeFactorize(i)
        pbar.update(1)
    pbar.close()
    return data

def is_pow_of_2(num):
    return np.mod(np.log(num)/np.log(2), 1)==0

def make_filename(lb, ub):
    if lb==2 and is_pow_of_2(ub):
        return '2^%d.json'%(int(np.log(ub)/np.log(2)))
    return '%d_%d.json'%(lb, ub)


def make_save_path(save_dir, lb, ub):
    os.makedirs(save_dir, exist_ok=True)
    fname = make_filename(lb, ub)
    return os.path.join(save_dir, fname)



def main(args):
    lb, ub = get_bounds(args)
    data = make_data(lb, ub)
    save_path = make_save_path(args.save_path, lb, ub)
    save_json(data, save_path)





if __name__ == "__main__":
    parser = ArgumentParser()
    # If min/max value is specified use that. Otherwise 2**(min/max power). Use 2 as lb if no min specified
    parser.add_argument('--max_power', type=int, help='max power of 2 to use', default=-1)
    parser.add_argument('--min_power', type=int, help='min power of 2 to use', default=-1)
    parser.add_argument('--max_value', type=int, help='max numebr to use', default=-1)
    parser.add_argument('--min_value', type=int, help='min number to use', default=2)
    parser.add_argument('--save_path', type=str, default='data/')

    args = parser.parse_args()
    main(args)