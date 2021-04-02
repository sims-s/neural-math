from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
# import torch.optim as optim
# from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append('./src/')
from utils import save_json, load_json, get_max_input_size, get_max_decode_size
from tokenizer import Tokenizer
from data_utils import FactorizationDataset, binarize_data
from models import Factorizer
from metrics_utils import form_factor_df, compute_metrics




def load_data_file(args):
    if args['data']['data_loc'].endswith('.json'):
        return load_json(args['data']['data_loc'])
    else:
        return load_json(args['data']['data_loc'] + '2^%d.json'%args['data']['max_pow'])

def get_datasets(args):
    if args['verbose']:
        print('Loading data...')
    data = load_data_file(args)
    data = binarize_data(data)
    
    train_dataset = FactorizationDataset(data['train'], args['data']['max_input_size'],
                                        args['data']['max_decode_size'],
                                        args['data']['input_padding'])
    test_dataset = FactorizationDataset(data['test'], args['data']['max_input_size'],
                                        args['data']['max_decode_size'],
                                        args['data']['input_padding'])
                                        
    train_loader = DataLoader(train_dataset, **args['loader']['train'])
    test_loader = DataLoader(test_dataset, **args['loader']['test'])

    if args['verbose']:
        print('Example data: ')
        print('\t' + str(train_dataset[0]))
        print('\t' + str(test_dataset[123]))
    
    return train_loader, test_loader
    


def get_model(args, device):
    model = Factorizer(n_tokens = args['tokenizer']['n_tokens'], 
                        pad_token_id = args['tokenizer']['pad_token_id'],
                        max_decode_size = args['data']['max_decode_size'],
                         **args['model_args'])
    model.to(device)
    return model


def load_model(args, device, state_dict):
    model = get_model(args, device)
    if args['verbose']:
        print('Successfully got model')
    model.load_state_dict(state_dict)
        
    return model

def get_just_test_data(args):
    return load_data_file(args)['test']

def compute_factorization_metrics(model, tokenizer, device, args):
    if args['verbose']:
        print('Computing metrics...')
    test_data = get_just_test_data(args)
    numbers = []
    for _, factorization_dict in test_data.items():
        numbers.append(factorization_dict['number'])
    if args['verbose']:
        print('Factoring...')
    metric_df = form_factor_df(model, tokenizer, device, numbers, args['data']['input_padding'], 
                               args['data']['max_input_size'], args['data']['max_decode_size'], 
                               args['metrics']['n_beams'], args['metrics']['max_num'])
    save_suffix = '_%s'%args['metrics']['suffix'] if len(args['metrics']['suffix']) > 0 else ''
    metric_df.to_csv(args['io']['save_path'] + 'metric_df%s.csv'%save_suffix)
    metric_df.to_pickle(args['io']['save_path'] + 'metric_df%s.pkl'%save_suffix)
    if args['verbose']:
        print('Computing metrics...')
    metrics = compute_metrics(metric_df)
    metrics['meta'] = {'n_beams' : args['metrics']['n_beams']}
    save_json(metrics, args['io']['save_path'] + 'metrics%s.json'%save_suffix)

def get_checkpoint(path):
    if path.endswith('.pt'):
        chosen_path = path
    else:
        files = [f for f in os.listdir(path) if f.endswith('.pt')]
        losses = [float(fname.split('_')[1][:-3]) for fname in files]
        best_loss_idx = np.argmin(losses)
        chosen_path = path + files[best_loss_idx]
    print('Loading model at %s'%chosen_path)
    return torch.load(chosen_path)
    
def update_args(args, input_args):
    args['metrics']['max_num'] = input_args.max_num
    args['metrics']['suffix'] = input_args.suffix
    return args


def main(input_args):
    checkpoint = get_checkpoint(input_args.path)
    args = checkpoint['args']
    # update the saved arguments incase you want to change stuff (e.g. # beams, etc)
    args = update_args(args, input_args)
    device = torch.device('cuda')
    tokenizer = Tokenizer()
    _, test_loader = get_datasets(args)
    model = load_model(args, device, checkpoint['model_state_dict'])
    compute_factorization_metrics(model, tokenizer, device, args)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--max_num', default=-1, type=int)
    parser.add_argument('--suffix', default='', type=str)
    args = parser.parse_args()
    main(args)


