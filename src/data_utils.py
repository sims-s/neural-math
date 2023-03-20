import numpy as np
import math
from numpy.lib.function_base import average
from sympy.matrices.dense import matrix_multiply_elementwise
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sympy import factorint, primepi, primerange, prime
from torch.utils.data.sampler import BatchSampler
import utils
from typing import List
import re
import copy

class RandomMemorylessSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, replacement = False,
                 num_samples = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        for i in range(n):
            yield np.random.randint(n)

    def __len__(self):
        return self.num_samples


class RandomMemorylessBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(self.sampler())
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore


def dec2base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def base2dec(num, base):
    total = 0
    for i, digit in enumerate(num[::-1]):
        if digit >= base:
            raise ValueError('Found token %d when trying to decode from base %d'%(digit, base))
        total += digit * (base ** i)
    return total















# def get_dataset(number_file, args, batch_size, is_train=False):
#     if args['model_type']=='addition':
#         return AdditionDataset(number_file, args['data']['base'])
#     elif args['model_type']=='factorization':
#         if number_file.endswith('.npy'):
#             return FactorizationDataset(number_file, args['data']['base'])
#         elif re.match("lossprop_\d+", number_file):
#             max_val = int(number_file.split('_')[1])
#             dataset_kwargs = args['data'].get('dataset_args', {})
#             return FactorizationDatasetPropLoss(max_val, 
#                                                 args['data']['base'], 
#                                                 test_vals = np.load(args['data']['test_path']),
#                                                 batch_size = batch_size,
#                                                 **dataset_kwargs)
#         elif re.match("pairwise_\d+", number_file):
#             max_val = int(number_file.split('_')[1])
#             dataset_kwargs = {'base' : args['data']['base']}
#             if is_train:
#                 dataset_kwargs['holdout_file'] = args['data']['holdout_file']
#             return FactorizationDatasetPairOFPrime(max_val, **dataset_kwargs)
#         else:
#             raise ValueError(f"number_file {number_file} not understood")



# def prepare_dataloader(number_file, args, is_train=False, **loader_kwargs):
#     batch_size = loader_kwargs.get('batch_size', 1)
#     data = get_dataset(number_file, args, batch_size, is_train=is_train)
#     sampler = None
#     if loader_kwargs.pop('random_sampling', False):
#         sampler = RandomMemorylessSampler(data)

#     loader = DataLoader(data, collate_fn = lambda x: {
#         'input': [y['input'] for y in x], 
#         'label' : [y['label'] for y in x]
#         },
#         sampler = sampler,
#         **loader_kwargs)
#     return loader