import numpy as np
import itertools
from argparse import ArgumentParser



def main(args):
    np.random.seed(42)
    data = itertools.product(np.arange(args.min_val, args.max_val + args.oos_size + 1), repeat=2)
    data = np.array(list(data))
    oos_indexer = np.max(data, axis=1) > args.max_val
    test_indexer = np.random.rand(data.shape[0]) < args.test_pct

    train_pairs = data[~test_indexer & ~oos_indexer]
    test_pairs = data[test_indexer & ~oos_indexer] 
    oos_pairs = data[oos_indexer]
    name_suffix = ''
    if args.min_val:
        name_suffix += str(args.min_val) + '-'
    name_suffix += str(args.max_val)


    for name, pair in zip(['train', 'test', 'oos'], [train_pairs, test_pairs, oos_pairs]):
        if pair.shape[0]:
            print(f'{name}, size: {pair.shape}, min: {pair.min()}, max: {pair.max()}')
            np.save(args.save_dir + f'{name}_data_{name_suffix}.npy', pair)
        else:
            print(f'{name} is empty!')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--max_val', required=True, type=int)
    parser.add_argument('--min_val', required=False, default=0, type=int)
    parser.add_argument('--test_pct', required=True, type=float)
    parser.add_argument('--oos_size', required=True, type=int)
    parser.add_argument('--save_dir', default='data/addition/')
    args = parser.parse_args()
    main(args)