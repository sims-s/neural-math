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
from data_utils import FactorizationDataset, binarize_data, prepare_dataloader
from models import Factorizer
from metrics_utils import compute_factorization_metrics
from utils import get_target_checkpoint, update_args_with_cli, backfill_args



def load_data_file(args):
    if args['data']['data_loc'].endswith('.json'):
        return load_json(args['data']['data_loc'])
    else:
        return load_json(args['data']['data_loc'] + '2^%d.json'%args['data']['max_pow'])

def get_test_dataset(args):
    if args['verbose']:
        print('Loading data...')
    data = load_data_file(args)
    return prepare_dataloader(data['test'], args, **args['loader']['test'])
    


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



def main(input_args):
    checkpoint = get_target_checkpoint(input_args.path)
    args = checkpoint['args']
    args = backfill_args(args)
    # update the saved arguments incase you want to change stuff (e.g. # beams, etc)
    args = update_args_with_cli(args, input_args)
    device = torch.device('cuda')
    tokenizer = Tokenizer()
    test_loader = get_test_dataset(args)
    model = load_model(args, device, checkpoint['model_state_dict'])
    compute_factorization_metrics(model, tokenizer, device, args)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--max_num', default=-1, type=int)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--n_beams', default=-1, type=int)
    args = parser.parse_args()
    main(args)


