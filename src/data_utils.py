import os
import re
import numpy as np
import utils
from torch.utils.data import Dataset, DataLoader


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

def form_label(label):
    to_join = []
    for _, factor_dict in label.items():
        to_join.append(utils.list_join([factor_dict['tokens']] * factor_dict['qty'], 'x'))
    factors = ['>'] + utils.list_join(to_join, 'x') + ['.']
    return factors

def form_input(number):
    return ['>'] + number + ['.']

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