import numpy as np
from argparse import ArgumentParser
from sympy import sieve




def main(args):
    if args.min_prime_factor <=1:
        numbers = np.arange(args.max_val)[2:]
    else:
        # first one is always 2 --> do that with np.arange
        primes_to_remove = list(sieve.primerange(args.min_prime_factor))[1:]
        numbers = np.arange(1, args.max_val, 2)[1:]
        for p in primes_to_remove:
            numbers = numbers[(numbers % p).astype(bool)]
    
    np.random.seed(args.seed)
    train_indexer = np.random.rand(numbers.shape[0]) > args.test_pct
    
    train_nums = numbers[train_indexer]
    test_nums = numbers[~train_indexer]

    save_path = args.save_dir + '%s_data'
    if args.suffix:
        save_path += '_%s'%args.suffix
    save_path += '.npy'
    print(train_nums.shape[0], test_nums.shape[0])
    np.save(save_path%'train', train_nums)
    np.save(save_path%'test', test_nums)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--max_val', type=int, default=2**32, help='genreate integers from 2...max_val')
    parser.add_argument('--min_prime_factor', type=int, default=1, help='Only save numbers that do not have primes smaller than this value as factors')
    parser.add_argument('--test_pct', type=float, default=.002)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='data/')
    parser.add_argument('--suffix', default='')


    args = parser.parse_args()
    main(args)