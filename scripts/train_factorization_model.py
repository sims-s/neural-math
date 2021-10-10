from argparse import ArgumentParser
from torch._C import Value
import yaml
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pprint
import sys
import wandb
sys.path.append('./src/')
from scheduler import get_linear_schedule_with_warmup
from tokenizer import Tokenizer
from data_utils import prepare_dataloader
from models import Factorizer
from optimization_utils import run_training
from metrics_utils import compute_factorization_metrics
from utils import get_best_checkpoint, get_last_checkpoint

def get_datasets(args):
    if args['verbose']:
        print('Loading data...')

    return prepare_dataloader(args['data']['train_path'], args, **args['loader']['train']), \
           prepare_dataloader(args['data']['test_path'], args, **args['loader']['test']), \
           prepare_dataloader(args['data']['oos_path'], args, **args['loader']['oos'])
    

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
    return getattr(optim, args['optimizer']['type'])(**args['optimizer']['opt_args'])
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

def compute_nb_steps(args, train_loader):
    args['scheduler']['nb_steps'] = args['scheduler']['nb_epochs'] * len(train_loader) // args['optimizer']['gradient_accumulation_steps']
    if args['scheduler']['max_steps'] > 0:
        args['scheduler']['nb_steps'] = min(args['scheduler']['nb_steps'], args['scheduler']['max_steps'])
    return args


def create_or_load_save_dir(args, model, optimizer, scheduler, map_location):
    checkpoint_dir = os.path.join(args['io']['save_path'], 'checkpoints')
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 1 and not args['resume_training']:
        raise ValueError("Save path %s already exists, but not resuming training. Add '--resume_training' to arguments to resume training"%args['io']['save_path'])
    elif args['resume_training']:
        latest_checkpoint = get_last_checkpoint(args['io']['save_path'], map_location)
        if latest_checkpoint:
            model.load_state_dict(latest_checkpoint['model_state_dict'])
            optimizer.load_state_dict(latest_checkpoint['opt_state_dict'])
            scheduler.load_state_dict(latest_checkpoint['scheduler_state_dict'])
    else:
        os.makedirs(args['io']['save_path'], exist_ok=True)
        os.makedirs(args['io']['save_path'] + 'checkpoints/', exist_ok=True)
        latest_checkpoint = None
    
    return model, optimizer, scheduler, latest_checkpoint


def wandb_init(args):
    if args['wandb']['enabled']:
        if 'id' in args['wandb']:
            wandb.init(project=args['wandb']['project'], entity=args['wandb']['entity'], 
                    id=args['wandb']['id'], config=args, resume=True)
        else:
            id = wandb.util.generate_id()
            wandb.init(project=args['wandb']['project'], entity=args['wandb']['entity'], id=id, config=args)
            args['wandb']['id'] = id

def search_dict(d, k_replace, v_replace):
    if isinstance(k_replace, str):
        k_replace = k_replace.split('.')    
    curr_dict = d
    found = True
    for i in range(len(k_replace)-1):
        check = k_replace[i]
        if check in curr_dict:
            curr_dict = curr_dict[check]
        else:
            found = False


    if found and k_replace[-1] in curr_dict:
        curr_dict[k_replace[-1]] = v_replace
        return

    for k, v in d.items():
        if isinstance(v, dict):
            search_dict(v, k_replace, v_replace)

def parse_unk_args(args, unknown):
    if len(unknown) % 2:
        raise ValueError('Didnt find an even number of arg/value pairs, got %d'%len(unknown))
    args_value_pair = [(unknown[2*i], unknown[2*i+1]) for i in range(len(unknown)//2)]
    for k_replace, v_replace in args_value_pair:
        print(k_replace, v_replace)
        search_dict(args, k_replace.replace('-', '').lower(), v_replace)
    return args

def main(args):
    device = torch.device('cuda')
    map_location = None

    tokenizer = Tokenizer(args['data']['base'])
    args = compute_extra_args(args, tokenizer)

    train_loader, test_loader, oos_loader = get_datasets(args)

    args = compute_nb_steps(args, train_loader)
    model, optimizer, scheduler, args = get_model_opt_scheduler(args, device)
    
    model, optimizer, scheduler, latest_checkpoint = create_or_load_save_dir(args, model, optimizer, scheduler, map_location)

    wandb_init(args)

    if args['verbose']:
        print('Running training for %d steps, %d warmup'%(args['scheduler']['nb_steps'], args['scheduler']['n_warmup_steps']))
        print('Model args: ')
        pprint.pprint(args['model_args'])
    run_training(model, optimizer, scheduler, tokenizer, train_loader, test_loader, oos_loader, device, args, latest_checkpoint)

    best_checkpoint = get_best_checkpoint(args['io']['save_path'], map_location=map_location)['model_state_dict']
    model.load_state_dict(best_checkpoint)
    compute_factorization_metrics(model, tokenizer, device, args, args['data']['test_path'], 
                save_suffix='test')
    compute_factorization_metrics(model, tokenizer, device, args, args['data']['oos_path'], 
                save_suffix = 'oos')


if __name__ == "__main__":
    """
    TODO: (no particular order)

    * Quick ideas to try:
        * Weight decay somewhere: RUN EXPERIMENT
        * AdamW vs Adam: RUN EXPERIMENT
        * Train the model for a lot longer
            * Use a different learning rate strategy

    * Write some documentation on how the arguments work
        * also cleanup docuemntation that's on github, becasue it's out of date

    * What if the model sees the factorization in multiple bases at the same time?
        * Diferent bases seem to perform differently, I wonder what combinding them would do

    """
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume_training', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)
        config_args['resume_training'] = args.resume_training
        config_args = parse_unk_args(config_args, unknown)

    
    main(config_args)


