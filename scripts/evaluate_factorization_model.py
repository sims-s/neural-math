from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

import sys
sys.path.append('./src/')
from utils import save_json, load_json
from tokenizer import Tokenizer
import data_utils
from data_utils import prepare_dataloader
from models import Factorizer
from metrics_utils import compute_factorization_metrics
from utils import get_best_checkpoint, update_args_with_cli, backfill_args


def get_model(args, device):
    model = Factorizer(n_tokens = args['tokenizer']['n_tokens'], 
                        pad_token_id = args['tokenizer']['pad_token_id'],
                         **args['model_args'])
    model.to(device)
    return model


def load_model(args, device, state_dict):
    model = get_model(args, device)
    if args['verbose']:
        print('Successfully got model')
    model.load_state_dict(state_dict)
        
    return model



def main(cli_args):
    checkpoint = get_best_checkpoint(cli_args.path)

    args = checkpoint['args']
    args = backfill_args(args)
    # update the saved arguments incase you want to change stuff (e.g. # beams, etc)
    args = update_args_with_cli(args, cli_args)

    device = torch.device('cuda')
    tokenizer = Tokenizer(args['data']['base'])
    model = load_model(args, device, checkpoint['model_state_dict'])
    compute_factorization_metrics(model, tokenizer, device, args, data_path=cli_args.data_path, save_suffix=cli_args.suffix)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--max_num', default=-1, type=int)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--n_beams', default=-1, type=int)
    parser.add_argument('--temperature', default=-1, type=float)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--num_workers', default=0, type=int, help='num workers for dataloader. default is one')
    args = parser.parse_args()
    main(args)


