from argparse import ArgumentParser
from tqdm.auto import tqdm
import os
import sys
import numpy as np
sys.path.append('./src/')
from utils import save_json
sys.path.append('./src/PrimeFactorization/')
from PrimeFactorization import primeFactorize


def make_data(lb, ub):
    data = {}
    pbar = tqdm(total = ub-lb, leave=False)
    for i in range(lb, ub):
        data[i] = primeFactorize(i)
        pbar.update(1)
    pbar.close()
    return data

def train_test_split(data, train_prob):
    train_data = {}
    test_data = {}
    train_counter = 0
    test_counter = 0

    np.random.seed(0)
    for number, factor in data.items():
        if np.random.rand() < train_prob:
            train_data[train_counter] = {'number' : number, 'factors' : factor} 
            train_counter +=1
        else:
            test_data[test_counter] = {'number' : number, 'factors' : factor} 
            test_counter +=1
    return {'train' : train_data, 'test' : test_data}

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
    ub = 2**args.max_power
    data = make_data(ub)
    data = train_test_split(data, args.train_prob)
    save_path = make_save_path(args.save_path, lb, ub)
    save_json(data, save_path)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--max_power', type=int, help='max power of 2 to use', default=-1)
    parser.add_argument('--save_folder', type=str, default='data/')
    parser.add_argument('--train_amt', type=float, default=.8, help='train size or percent. If in [0,1] used as percent. Otherwise quantity')

    args = parser.parse_args()
    main(args)