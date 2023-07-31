import numpy as np
from argparse import ArgumentParser
from sympy import primepi, factorint
from tqdm.auto import tqdm

def main(args):
    primes = np.load(args.primes_file, mmap_mode='r')
    numbers = []
    max_prime_idx = primepi(args.max_val // 2)
    sample_range = np.arange(args.min_prime_idx, max_prime_idx)
    if args.incorporate_other_test_set:
        for num in np.load(args.incorporate_other_test_set):
            factorization = factorint(num, multiple=True)
            if len(factorization)==2:
                numbers.append(num)

    skip_nums = set()
    if args.skip_file:
        for f in args.skip_file.split(','):
            skip_nums.update(set(np.load(f).tolist()))
        print(f'skip {len(skip_nums)} numbers')
            

    if args.just_range:
        lb,ub = [int(num.strip()) for num in args.just_range.split(',')]
        for num in np.arange(lb, ub):
            if num in skip_nums:
                continue
            factorization = factorint(num, multiple=True)
            if len(factorization)==2:
                numbers.append(num)
    else:
        run_size = len(numbers) - args.size
        with tqdm(total = run_size) as pbar:
            for _ in range(run_size):
                first_prime = primes[np.random.choice(sample_range)]
                remainder = args.max_val // first_prime
                other_max_prime_idx = primepi(remainder)
                second_prime = primes[np.random.choice(other_max_prime_idx)]
                product = first_prime * second_prime
                numbers.append(product)
                pbar.update(1)

    print('size: ', len(numbers))
    np.save(args.save_path, np.array(numbers))


if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--size', type=int)
    parser.add_argument('--max_val', type=int, default=-1)
    parser.add_argument('--min_prime_idx', required=False, default=0, type=int)
    parser.add_argument('--primes_file', required=False, default='data/primes_2B.npy')
    parser.add_argument('--incorporate_other_test_set', default='', required=False)
    parser.add_argument('--just_range', default='', required=False)
    parser.add_argument('--skip_file', default='')

    args = parser.parse_args()
    main(args)