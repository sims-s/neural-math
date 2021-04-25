import os
import re
import numpy as np
import utils
from torch.utils.data import Dataset, DataLoader

def dec2bin(num):
    return bin(int(num))[2:]

def bin2dec(num):
    return int(str(num), 2)

def binarize_data(data):
    if 'train' in data:
        data['train'] = binarize_data(data['train'])
        data['test'] = binarize_data(data['test'])
        return data
    binarized = {}
    for i, factor_dict in data.items():
        binarized[i] = {'number' : dec2bin(factor_dict['number']), 'factors' : {dec2bin(int(factor)) : qty for factor, qty in factor_dict['factors'].items()}}
    return binarized


def prepare_dataloader(data, args, **loader_kwargs):
    data = binarize_data(data)
    data = FactorizationDataset(data, args['data']['max_input_size'],
                                        args['data']['max_decode_size'],
                                        args['data']['input_padding'])
    loader = DataLoader(data, **loader_kwargs)
    return loader


def pad_input(input, pad_to, input_padding):
    n_pad = pad_to - len(input)
    if input_padding=='pad':
        return input + '.' + '_'*n_pad
    elif input_padding=='zeros':
        return '0'*n_pad + input
    else:
        raise ValueError()

class FactorizationDataset(Dataset):
    def __init__(self, data_dict, max_encode_size, max_decode_size, input_padding):
        self.data_dict = self.keys_to_int(data_dict)
        self.max_encode_size = max_encode_size
        self.max_decode_size = max_decode_size
        self.input_padding = input_padding

    def keys_to_int(self, data):
        new_data = {}
        for k, v in data.items():
            new_data[int(k)] = v
        return new_data
    
    def form_label(self, label):
        factors = sum([[k]*v for k, v in label.items()], [])
        factors = 'x'.join(factors) + '.'
        n_pad = self.max_decode_size - len(factors)
        factors = factors + '_'*n_pad
        return factors
    
    
    def __getitem__(self, i):
        number = self.data_dict[i]['number']
        label = self.data_dict[i]['factors']
        return {'number' : pad_input(number, self.max_encode_size, self.input_padding), 'label' : self.form_label(label)}
        
    def __len__(self):
        return len(self.data_dict)


class GlobalFactorMapping():
    def __init__(self, data_path = 'data/2^16.json'):
        if data_path.endswith('.json'):
            best_path = data_path
        else:
            name_pattern = re.compile('2\^\d+')
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