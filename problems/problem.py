import numpy as np
import pandas as pd
import sys
sys.path.append('./src/')
from tokenizer import Tokenizer
import data_utils
import utils
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import torch
from generation_utils import decode
from optimization_utils import test_on_loader

class Problem:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None

    def get_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = Tokenizer(self.args['data']['base'])
            self.update_tokenizer_args(self, self.tokenizer)
        return self.tokenizer

    def update_tokenizer_args(self, args, tokenizer):
        # # Compute input/output sizes given how we're padding things
        # Return things related to the tokenizer
        self.args['tokenizer'] = {}
        self.args['tokenizer']['n_tokens'] = len(tokenizer)
        self.args['tokenizer']['pad_token_id'] = tokenizer.encode('_')[0]

    def get_dataset(self):
        raise NotImplementedError

    def postprocess(self, output, logprob, beam_idx, tokenizer):
        raise NotImplementedError

    # def form_input(self, [args_for_problem]):
    #     raise NotImplementedError

    # def form_label(self, [args_for_problem]):
    #     raise NotImplementedError



    def prepare_dataloader(self, path_or_dataset, **loader_kwargs):
        if isinstance(path_or_dataset, str):
            dataset = self.get_dataset(path_or_dataset)
        else:
            dataset = path_or_dataset

        sampler = None
        if loader_kwargs.pop('random_sampling', False):
            sampler = data_utils.RandomMemorylessSampler(dataset)

        loader = DataLoader(dataset, collate_fn = lambda x: {
            'input': [y['input'] for y in x], 
            'label' : [y['label'] for y in x]
            },
            sampler = sampler,
            **loader_kwargs)
        return loader


    def compute_metrics(self, model, device, path_or_dataset, save_dir=None, save_suffix='', loader_kwargs=None, **kwargs):
        if loader_kwargs is None:
            loader_kwargs = {}
        if isinstance(path_or_dataset, str):
            dataset = self.get_dataset(path_or_dataset)
        else:
            dataset = path_or_dataset
        args = self.args
        prediction_df = self.form_prediction_df(model, device, dataset, args['model_args']['max_decode_size'],
                               args['metrics']['n_beams'], args['metrics']['temperature'], args['metrics']['max_num'], **kwargs)

        if not save_dir:
            if 'save_path' in args['io']:
                save_dir = args['io']['save_path']
        if save_suffix:
            save_name = f'{save_dir}/pred_df_{save_suffix}.csv'
        else:
            save_name = f'{save_dir}/pred_df.csv'
        
        prediction_df.to_csv(save_name, index=False)

        metrics = self.aggregate_metrics(prediction_df, save_dir=save_dir, save_suffix=save_suffix)

        loader = self.prepare_dataloader(dataset, **loader_kwargs)
        metrics['loss'] = test_on_loader(model, loader, self.tokenizer, nn.CrossEntropyLoss(), device)
        metrics['meta'] = {'n_beams' : args['metrics']['n_beams'], 'temperature' : args['metrics']['temperature']}

        if save_suffix:
            save_name = f'{save_dir}/metrics_{save_suffix}.json'
        else:
            save_name = f'{save_dir}/metrics.json'
        utils.save_json(metrics, save_name)



    def form_prediction_df(self, model, device, dataset, max_decode_size=64, n_beams=1, temperature=1, max_num=-1, **kwargs):
        rows = []    
        pbar = tqdm(total = min(len(dataset), max_num) if max_num > 0 else len(dataset), leave=False)
        for i in range(len(dataset)):
            datum = dataset[i]
            if max_num > 0 and i >= max_num:
                break
            rows.append(self.solve(datum, model, device, max_decode_size, n_beams, temperature, return_type='dict', **kwargs))
            pbar.update(1)
        pbar.close()
        rows = sum(rows, [])
        df = pd.DataFrame.from_dict(rows)
        return df

    def solve(self, datum, model, device, max_decode_size, n_beams, temperature, return_type='dict', **kwargs):
        if isinstance(datum, dict) and 'input' in datum:
            input = datum['input']
        else:
            if not isinstance(datum, list):
                datum = [datum]
            input = self.form_input(*datum, base=self.args['data']['base'])
        untokenized_input = input
        input = torch.tensor(self.tokenizer.encode(input), device=device).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            memory, memory_key_padding_mask = model.encode(input)
        outputs, logprobs = decode(model, self.tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams, temperature)

        to_return = []
        for i in range(len(outputs)):
            to_return.append(self.postprocess(outputs[i], logprobs[i], i, untokenized_input, **kwargs))

        if return_type=='df':
            return pd.DataFrame.from_dict(to_return)
        elif return_type=='dict':
            return to_return
        else:
            raise ValueError('got unexpected return type %s'%return_type)



class BaseDataset(Dataset):
    def __init__(self, number_file, base, form_input, form_label):
        self.numbers = np.load(number_file, mmap_mode='r')
        self.base = base
        self.form_input = form_input
        self.form_label = form_label

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return self.numbers.shape[0]