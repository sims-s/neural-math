import numpy as np
from argparse import ArgumentParser
from sympy import sieve




def main(args):
    numbers = np.arange(args.max_val)[2:]
    
    np.random.seed(args.seed)
    train_indexer = np.random.rand(numbers.shape[0]) > args.test_pct
    
    train_nums = numbers[train_indexer]
    test_nums = numbers[~train_indexer]

    save_path = args.save_dir + '%s_data'
    if args.suffix:
        save_path += '_%s'%args.suffix
    save_path += '.npy'
    print('Train/test size: ', train_nums.shape[0], test_nums.shape[0])
    np.save(save_path%'train', train_nums)
    np.save(save_path%'test', test_nums)

    if args.oos_size >=1:
        oos_nums = np.arange(args.max_val, args.max_val + args.oos_size)
        print('OoS size: ', oos_nums.shape[0])
        np.save(save_path%'oos', oos_nums)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--max_val', type=int, default=2**18, help='genreate integers from 2...max_val')
    parser.add_argument('--oos_size', type=int, default=1024, help='generate an additional out of sample dataset from [max_val, max_val + higher_set_size] (i.e all numbers higher than models training size)')
    parser.add_argument('--test_pct', type=float, default=.1)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='data/')
    parser.add_argument('--suffix', default='')


    args = parser.parse_args()
    main(args)