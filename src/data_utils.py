import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sympy import factorint
import utils

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
        yield np.random.randint(n)

    def __len__(self):
        return self.num_samples


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



def form_label(number, base):
    factors = factorint(number, multiple=True)
    factors = [dec2base(f, base) for f in factors]
    factors = ['>'] + utils.list_join(factors, 'x') + ['.']
    return factors

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

def prepare_dataloader(number_file, args, **loader_kwargs):
    data = FactorizationDataset(number_file, args['data']['base'])
    sampler = None
    if loader_kwargs.pop('random_sampling', False):
        pass
    #     sampler = RandomMemorylessSampler(data)
    # print(sampler)
    # sys.exit()
    loader = DataLoader(data, collate_fn = lambda x: {
        'number': [y['number'] for y in x], 
        'label' : [y['label'] for y in x]},
        sampler = sampler,
        **loader_kwargs)
    return loader



"""
class FactorizationDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = self.keys_to_int(data_dict)

    def keys_to_int(self, data):
        new_data = {}
        for k, v in data.items():
            new_data[int(k)] = v
        return new_data
    
    def __getitem__(self, i):
        number = self.data_dict[i]['number']
        label = self.data_dict[i]['factors']
        return {'number' : form_input(number), 'label' : form_label(label)}
        
    def __len__(self):
        return len(self.data_dict)


class GlobalFactorMapping():
    def __init__(self, data_path = 'data/2^16.json'):
        if data_path.endswith('.json'):
            best_path = data_path
        else:
            name_pattern = re.compile(r'2\^\d+')
            files = os.listdir(data_path)
            files = [f for f in files if f.endswith('.json') and name_pattern.search(f)]
            powers = [int(f.split('.')[0].split('^')[1]) for f in files]
            best_idx = np.argmax(powers)
            best_path = data_path + files[best_idx]
        
        self.best_path = best_path
        self.mapper = None

    def initialize(self):
        dataset = utils.load_json(self.best_path)
        final_mapper = {}
        for train_or_test, mapper in dataset.items():
            for _, factor_dict in mapper.items():
                final_mapper[int(factor_dict['number'])] = {int(factor) : int(qty) for factor, qty in factor_dict['factors'].items()}
        self.mapper = final_mapper
            
    def __contains__(self, item):
        return item in self.mapper

    def __call__(self, i):
        return self[i]

    def __getitem__(self, i):
        if not self.mapper:
            self.initialize()
        return self.mapper[i]

    def is_prime(self, num):
        try:
            factored = self[num]
        except KeyError:
            return None
        return len(factored)==1 and factored[list(factored.keys())[0]]==1

gfm = None


def convert_base(data, base):
    if 'train' in data:
        data['train'] = convert_base(data['train'], base)
        data['test'] = convert_base(data['test'], base)
        return data
    converted = {}
    for i, factor_dict in data.items():
        converted[i] = {'number' : dec2base(factor_dict['number'], base), 
                        'factors' : {
                            i : {
                                    'tokens' : dec2base(int(factor), base),
                                    'qty' : qty,
                                }
                                for i, (factor, qty) in enumerate(factor_dict['factors'].items())}
                        }
    return converted


def prepare_dataloader(data, args, **loader_kwargs):
    data = convert_base(data, args['data']['base'])
    data = FactorizationDataset(data)
    loader = DataLoader(data, collate_fn = lambda x: {
        'number': [y['number'] for y in x], 
        'label' : [y['label'] for y in x]}, 
        **loader_kwargs)
    return loader

"""