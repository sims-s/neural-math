import numpy as np
from argparse import ArgumentParser
from sympy import factorint
from tqdm.auto import tqdm


def make_factorization_array(numbers):
    factors = [factorint(num, multiple=True) for num in tqdm(numbers, total=len(numbers))]
    max_len = max([len(f) for f in factors])
    arr = np.ones((len(numbers), max_len))
    for i, f in enumerate(factors):
        arr[i, :len(f)] = f
    return arr.astype(np.uint32)

def main(args):
    numbers = np.arange(args.max_val)[2:]
    
    np.random.seed(args.seed)
    train_indexer = np.random.rand(numbers.shape[0]) > args.test_pct
    
    train_nums = numbers[train_indexer]
    test_nums = numbers[~train_indexer]

    train_factors = make_factorization_array(train_nums)
    test_factors = make_factorization_array(test_nums)

    save_path = args.save_dir + '%s_data'
    if args.suffix:
        save_path += '_%s'%args.suffix
    save_path += '.npy'
    print('Train/test size: ', train_nums.shape[0], test_nums.shape[0])
    print(train_factors.dtype)
    np.save(save_path%'train', train_factors)
    np.save(save_path%'test', test_factors)

    if args.oos_size >=1:
        oos_nums = np.arange(args.max_val, args.max_val + args.oos_size)
        oos_factors = make_factorization_array(oos_nums)
        print('OoS size: ', oos_nums.shape[0])
        np.save(save_path%'oos', oos_factors)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--max_val', type=int, default=2**18, help='genreate integers from 2...max_val')
    parser.add_argument('--oos_size', type=int, default=2048, help='generate an additional out of sample dataset from [max_val, max_val + higher_set_size] (i.e all numbers higher than models training size)')
    parser.add_argument('--test_pct', type=float, default=.1)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='data/')
    parser.add_argument('--suffix', default='')


    args = parser.parse_args()
    main(args)