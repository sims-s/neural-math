from argparse import ArgumentParser
import yaml
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append('./src/')
from utils import save_json, load_json, load_data_file
from tokenizer import Tokenizer
import data_utils
from data_utils import prepare_dataloader, GlobalFactorMapping, gfm
from models import Factorizer
from optimization_utils import run_training
from generation_utils import factor
from metrics_utils import compute_factorization_metrics
from utils import get_best_checkpoint, backfill_args



def get_datasets(args):
    if args['verbose']:
        print('Loading data...')
    data = load_data_file(args['data']['data_loc'])
    
    return prepare_dataloader(data['train'], args, **args['loader']['train']), \
           prepare_dataloader(data['test'], args, **args['loader']['test'])
    

def compute_extra_args(args, tokenizer):
    # # Compute input/output sizes given how we're padding things
    # Return things related to the tokenizer
    args['tokenizer'] = {}
    args['tokenizer']['n_tokens'] = len(tokenizer)
    args['tokenizer']['pad_token_id'] = tokenizer.encode('_')[0]
    return args


def get_model(args, device):
    model = Factorizer(n_tokens = args['tokenizer']['n_tokens'], 
                        pad_token_id = args['tokenizer']['pad_token_id'],
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



def main(args):
    device = torch.device('cuda')
    tokenizer = Tokenizer(args['data']['base'])
    args = compute_extra_args(args, tokenizer)
    data_utils.gfm = GlobalFactorMapping(data_path = args['data']['data_loc'] if args['data']['data_loc'].endswith('.json') else \
                                          args['data']['data_loc'] + '2^%d.json'%args['data']['max_pow'])



    train_loader, test_loader = get_datasets(args)
    args['scheduler']['nb_steps'] = args['scheduler']['nb_epochs'] * len(train_loader)
    
    os.makedirs(args['io']['save_path'], exist_ok=True)
    os.makedirs(args['io']['save_path'] + 'checkpoints/', exist_ok=True)

    model, optimizer, scheduler, args = get_model_opt_scheduler(args, device)

    if args['verbose']:
        print('Running training for %d steps, %d warmup'%(args['scheduler']['nb_steps'], args['scheduler']['n_warmup_steps']))
        print('Model args: ', args['model_args'])

    run_training(model, optimizer, scheduler, tokenizer, train_loader, test_loader, device, args)
    
    best_checkpoint = get_best_checkpoint(args['io']['save_path'])['model_state_dict']
    model.load_state_dict(best_checkpoint)
    compute_factorization_metrics(model, tokenizer, device, args)

    # Training rework:
        # Ability to resume training
        # Checkpoint at not end of epoch
        # Gradient Accumulation
    
    # Why are real tokens predicted after padding? That shoulda been shut down.... FIx that bug!!

    # Scale up to generalized bases

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        args = yaml.safe_load(f)
    main(args)


