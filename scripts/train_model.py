from argparse import ArgumentParser
import yaml
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append('./src/')
from utils import save_json, load_json, get_max_input_size, get_max_decode_size
from tokenizer import Tokenizer
from data_utils import FactorizationDataset, binarize_data
from models import Factorizer
from optimization_utils import run_training
from generation_utils import factor
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
    

def compute_extra_args(args, tokenizer):
    # Compute input/output sizes given how we're padding things
    args['data']['max_input_size'] = get_max_input_size(args['data']['max_pow'], args['data']['input_padding'])
    args['data']['max_decode_size'] = get_max_decode_size(args['data']['max_pow'])
    # Return things related to the tokenizer
    args['tokenizer'] = {}
    args['tokenizer']['n_tokens'] = len(tokenizer)
    args['tokenizer']['pad_token_id'] = tokenizer('_')[0]
    return args


def get_model(args, device):
    model = Factorizer(n_tokens = args['tokenizer']['n_tokens'], 
                        pad_token_id = args['tokenizer']['pad_token_id'],
                        max_decode_size = args['data']['max_decode_size'],
                         **args['model_args'])
    model.to(device)
    return model

def get_optimizer(args, model):
    if args['optimizer']['type'].lower()=='adam':
        opt = optim.Adam(model.parameters(), **args['optimizer']['opt_args'])
    else:
        raise ValueError('Only using adam right now')
    return opt

def get_scheduler(args, optimizer):
    # linear_schedule_with_warmup
    if args['scheduler']['type']=='linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, args['scheduler']['n_warmup_steps'], args['scheduler']['nb_steps'])
    else:
        raise ValueError('only using linear_schedule_with_warmup right now')
    return scheduler


def get_model_opt_scheduler(args, device):
    model = get_model(args, device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    if args['verbose']:
        print('Successfully created model, optimizer, and scheduler')
        
    return model, optimizer, scheduler, args

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
    metric_df.to_csv(args['io']['save_path'] + 'metric_df.csv')
    metric_df.to_pickle(args['io']['save_path'] + 'metric_df.pkl')
    if args['verbose']:
        print('Computing metrics...')
    metrics = compute_metrics(metric_df)
    save_json(metrics, args['io']['save_path'] + 'metrics.json')



def main(args):
    device = torch.device('cuda')
    tokenizer = Tokenizer()
    args = compute_extra_args(args, tokenizer)
    train_loader, test_loader = get_datasets(args)
    args['scheduler']['nb_steps'] = args['scheduler']['nb_epochs'] * len(train_loader)
    os.makedirs(args['io']['save_path'], exist_ok=True)
    model, optimizer, scheduler, args = get_model_opt_scheduler(args, device)
    run_training(model, optimizer, scheduler, tokenizer, train_loader, test_loader, device, args)
    compute_factorization_metrics(model, tokenizer, device, args)
    
    # Write a script for doing evaluation
    
    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        args = yaml.safe_load(f)
    main(args)


