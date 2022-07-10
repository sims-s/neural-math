import numpy as np
import math
from numpy.lib.function_base import average
from sympy.matrices.dense import matrix_multiply_elementwise
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sympy import factorint, primepi, primerange, prime
from torch.utils.data.sampler import BatchSampler
import utils
from torch._six import int_classes as _int_classes
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
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
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

def form_label_from_factorlist(factors, base):
    factors = [dec2base(f, base) for f in factors]
    factors = ['>'] + utils.list_join(factors, 'x') + ['.']
    return factors


def form_label(number, base):
    factors = factorint(number, multiple=True)
    return form_label_from_factorlist(factors, base)
    

def form_input(number, base):
    number = dec2base(number, base)
    return ['>'] + number + ['.']



class FactorizationDataset(Dataset):
    def __init__(self, number_file, base):
        self.numbers = np.load(number_file, mmap_mode='r')
        self.base = base
    
    def __getitem__(self, i):
        number = self.numbers[i]
        return {
            'number' : form_input(number, self.base),
            'label' : form_label(number, self.base)
        }
    def __len__(self):
        return self.numbers.shape[0]


def form_input_addition(n1, n2, base):
    return ['>'] + dec2base(n1, base) + ['+'] + dec2base(n2, base) + ['.']

def form_label_addition(n1, n2, base):
    return ['>'] + dec2base(n1 + n2, base) + ['.']


class AdditionDataset(Dataset):
    def __init__(self, number_file, base):
        self.numbers = np.load(number_file, mmap_mode='r')
        self.base = base

    def __getitem__(self, i):
        n1, n2 = self.numbers[i]
        output =  {
            'number' : form_input_addition(n1, n2, self.base),
            'label' : form_label_addition(n1, n2, self.base)
        }
        return output

    def __len__(self):
        return self.numbers.shape[0]

class FactorizationDatasetPairOFPrime(Dataset):
    def __init__(self, max_val, base, primes_file='data/primes_2B.npy', holdout_file = None):
        self.max_val = max_val
        self.max_prime_idx = int(primepi(max_val//2))
        self.length = int(max_val * .2)
        self.primes = np.load(primes_file, mmap_mode='r')
        self.max_val = self.primes[self.max_prime_idx] * 2
        self.base = base
        if isinstance(holdout_file, str):
            holdout_nums = np.load(holdout_file).tolist()
        else:
            holdout_nums = []
            for f in holdout_file:
                holdout_nums.extend(np.load(f).tolist())
        self.holdout_set = set(holdout_nums)
        
    def __getitem__(self, i):
        first_prime = self.primes[np.random.choice(self.max_prime_idx)]
        remainder = self.max_val // first_prime
        other_max_prime_idx = primepi(remainder)
        second_prime = self.primes[np.random.choice(other_max_prime_idx)]
        product = first_prime * second_prime
        first_prime, second_prime = min(first_prime, second_prime), max(first_prime, second_prime)
        if product in self.holdout_set:
            return self.__getitem__(0)

        return {
            'number' : form_input(product, self.base),
            'label' : form_label_from_factorlist([first_prime, second_prime], self.base)
        }
        
    def __len__(self):
        return self.length


class FactorizationDatasetPropLoss(Dataset):
    """
    Sampling Method for generating number X
    1. Sample the number of unique primes that X will have
    2. For each unique prime sample a bucket of common maximum multiplicitiy:
        All numbers in that bucket y can have the same maximum multplicitiy
        i.e. y^m <= max_val, where m is the multplicity
    3. Sample a number in the selected bucket
    4. If the number sampled is above the maximum value, resample
    5. When training the model, keep track of the mean loss across all 3 sampling steps:
        mean loss for each number of unique prime factors
        mean loss for each multiplicity group
        each loss for all primes in that multiplicity group
    6. When sampling in 1, 2, 3 sample in a way that is weighted toward the mean loss:
        a bucket with higher loss is more likely to be sampled
    """
    def __init__(self, max_val, base, test_vals, batch_size, loss_initialization = 1, len_replacement = 20000,
                        ema_constant = .99, primes_file='data/primes_2B.npy'):
        self.max_val = max_val
        self.base = base
        self.test_vals = set(list(test_vals))
        self.batch_size = batch_size
        self.loss_initialization = loss_initialization
        self.max_multiplicity = math.floor(math.log2(self.max_val))
        self.ema_constant = ema_constant
        self.len_replacement = len_replacement
        self.primes = np.load(primes_file, mmap_mode='r')
        self.setup_sampling()
        self.nunique_primes_historical = []
        self.multiplicity_historical = []
        self.prime_indicies_historical = []


    def setup_nunique_prime_weights(self):
        max_n_unique = 1
        while np.prod(list(primerange(2, prime(max_n_unique)))) < self.max_val:
            max_n_unique+=1
        max_n_unique -=1
        self.nunique_prime_weights = self.loss_initialization * np.ones(max_n_unique)

    def setup_multiplicity_weights(self):
        self.multiplicity_weights = self.loss_initialization * np.ones(self.max_multiplicity - 1)

    def setup_number_weights(self):
        self.multiplicity_number_weights = {
            m: self.loss_initialization * np.ones(primepi(self.max_val ** (1/m))) for m in range(1, self.max_multiplicity)
        }
        self.multiplicity_weight_scaling = np.array([loss_initialization.shape[0] for _, loss_initialization in self.multiplicity_number_weights.items()])

    def sample_array(self, arr):
        return np.random.choice(np.arange(len(arr)), p = arr / arr.sum())

    def setup_sampling(self):
        self.setup_nunique_prime_weights()
        self.setup_multiplicity_weights()
        self.setup_number_weights()

        # print('nunique prime weights shape: ', self.nunique_prime_weights.shape)
        # print('multiplicitiy weights shape: ', self.multiplicity_weights.shape)
        # print('multiplicity number weights shape: ')
        # for multiplicity, matrix in self.multiplicity_number_weights.items():
        #     print(multiplicity, matrix.shape)


    def sample_number(self, times_called = 0):
        nunique_primes = self.sample_array(self.nunique_prime_weights)+1
        # print('finish sampling nunique primes')
        multiplicities = [self.sample_array(self.multiplicity_weights * self.multiplicity_weight_scaling)+1 for _ in range(nunique_primes)]
        # print('initially sample multiplicities')
        # fail_counter = 0

        # easy heuristic to filter out invalid numbers
        if np.sum(multiplicities) > self.max_multiplicity:
            return self.sample_number(times_called+1)
            multiplicities = [self.sample_array(self.multiplicity_weights)+1 for _ in range(nunique_primes)]
            fail_counter+=1
        # print(f'failed {fail_counter} times')
        # print('finish smapling multplicities')
        prime_indicies = [self.sample_array(self.multiplicity_number_weights[m]) for m in multiplicities]
        # print('finish sampling prime indicies')

        factors = self.primes[prime_indicies]
        number = np.prod(factors)
        # number < 0 to handle overflow errors....
        if number > self.max_val or number < 0:
            return self.sample_number(times_called+1)
        factors.sort()

        # Keep track of what we've sampled so we know how to update the weights easily given the loss
        self.nunique_primes_historical.append(nunique_primes)
        self.multiplicity_historical.append(multiplicities)
        self.prime_indicies_historical.append(prime_indicies)


        # print(f'Sampled {times_called} before picking a number')
        # print('number: ', number)
        # print('factors: ', factors)
        # print('multiplicities: ', multiplicities)
        # print('='*100)
        return number, factors

    

    def __getitem__(self, i):
        number, factors = self.sample_number()
        return {
            'number' : form_input(number, self.base),
            'label' : form_label_from_factorlist(factors, self.base)
        }

    def update_nunique_weights(self, per_item_loss):
        self.nunique_primes_historical = np.array(self.nunique_primes_historical)
        assert len(self.nunique_primes_historical)==per_item_loss.shape[0]
        for val in self.nunique_primes_historical.unique():
            indexer = self.nunique_primes_historical==val
            self.nunique_prime_weights[val-1] = (self.nunique_prime_weights[val-1] * self.ema_constant) + \
                                                (per_item_loss[indexer].mean().item() * (1-self.ema_constant))
        self.nunique_primes_historical = []

    def update_multiplicity_weights(self, per_item_loss):
        assert len(self.multiplicity_historical)==per_item_loss.shape[0]
        for val in np.hstack(self.multiplicity_historical).unique():
            average_weights = [np.mean(np.array(multiplicity_list)==val) for multiplicity_list in self.multiplicity_historical]
            self.multiplicity_weights[val-1] = (self.multiplicity_weights[val-1]*self.ema_constant) + \
                                               (np.average(per_item_loss, weights=average_weights)) * (1-self.ema_constant)

        self.multiplicity_historical = []

    def update_number_multiplicity_weights(self, per_item_loss):
        assert len(self.multiplicity_historical)==len(self.prime_indicies_historical)==per_item_loss.shape[0]

        n_primes_historical = np.array([[i]*len(multiplicity_list) for i, multiplicity_list in enumerate(self.multiplicity_historical)])
        stacked = np.vstack(
            np.hstack(self.multiplicity_historical),
            np.hstack(self.prime_indicies_historical),
            np.hstack(n_primes_historical)
        )
        update_vals = np.unique(stacked[:,:2], axis=0)
        for update_pair in update_vals:
            sample_indicies = (stacked[:,:2]==update_pair).all(axis=1)
            self.multiplicity_number_weights[update_pair[0]][update_pair[1]] = (self.multiplicity_number_weights[update_pair[0]][update_pair[1]] * self.ema_constant) + \
                                                                               (per_item_loss[sample_indicies].mean() * (1 - self.ema_constant))
        self.prime_indicies_historical = []

    
        
    def update_weights(self, per_item_loss):
        per_item_loss = per_item_loss.data.cpu().numpy()
        self.update_number_multiplicity_weights(self, per_item_loss)
        self.update_multiplicity_weights(self, per_item_loss)
        self.update_nunique_weights(per_item_loss)

        
    def __len__(self):
        return self.len_replacement

def get_dataset(number_file, args, batch_size, is_train=False):
    if args['model_type']=='addition':
        return AdditionDataset(number_file, args['data']['base'])
    elif args['model_type']=='factorization':
        if number_file.endswith('.npy'):
            return FactorizationDataset(number_file, args['data']['base'])
        elif re.match("lossprop_\d+", number_file):
            max_val = int(number_file.split('_')[1])
            dataset_kwargs = args['data'].get('dataset_args', {})
            return FactorizationDatasetPropLoss(max_val, 
                                                args['data']['base'], 
                                                test_vals = np.load(args['data']['test_path']),
                                                batch_size = batch_size,
                                                **dataset_kwargs)
        elif re.match("pairwise_\d+", number_file):
            max_val = int(number_file.split('_')[1])
            dataset_kwargs = {'base' : args['data']['base']}
            if is_train:
                dataset_kwargs['holdout_file'] = args['data']['holdout_file']
            return FactorizationDatasetPairOFPrime(max_val, **dataset_kwargs)
        else:
            raise ValueError(f"number_file {number_file} not understood")



def prepare_dataloader(number_file, args, is_train=False, **loader_kwargs):
    batch_size = loader_kwargs.get('batch_size', 1)
    data = get_dataset(number_file, args, batch_size, is_train=is_train)
    sampler = None
    if loader_kwargs.pop('random_sampling', False):
        sampler = RandomMemorylessSampler(data)

    loader = DataLoader(data, collate_fn = lambda x: {
        'number': [y['number'] for y in x], 
        'label' : [y['label'] for y in x]
        },
        sampler = sampler,
        **loader_kwargs)
    return loader