from argparse import ArgumentParser
import yaml
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pprint
import sys
# import wandb
sys.path.append('./src/')
from tokenizer import Tokenizer
from data_utils import prepare_dataloader
from models import Factorizer
from optimization_utils import run_training, get_scheduler
from metrics_utils import compute_factorization_metrics
from utils import get_best_checkpoint, get_last_checkpoint

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


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
    args['multi_gpu'] = torch.cuda.device_count() > 1 and args['multi_gpu']
    return args


def get_model(args, device):
    model = Factorizer(n_tokens = args['tokenizer']['n_tokens'], 
                        pad_token_id = args['tokenizer']['pad_token_id'],
                         **args['model_args'])
    model.to(device)
    if args['multi_gpu']:
        model = DDP(model, device_ids=[device])
    return model

def get_optimizer(args, model):
    if args['optimizer']['type'].lower()=='adam':
        opt = optim.Adam(model.parameters(), **args['optimizer']['opt_args'])
    else:
        raise ValueError('Only using adam right now')
    return opt


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


def wandb_init(args):
    if args['wandb']['enabled']:
        if 'id' in args['wandb']:
            wandb.init(project=args['wandb']['project'], entity=args['wandb']['entity'], 
                    id=args['wandb']['id'], config=args, resume=True)
        else:
            id = wandb.util.generate_id()
            wandb.init(project=args['wandb']['project'], entity=args['wandb']['entity'], id=id, config=args)
            args['wandb']['id'] = id
        
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(f, world_size, args):
    mp.spawn(f, args=(world_size, args), nprocs=world_size, join=True)

def main(rank, args):   
    if torch.cuda.device_count() > 1 and args['multi_gpu']:
        setup(rank, torch.cuda.device_count())
        device = rank
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    else:
        device = torch.device('cuda')
        map_location = None
    tokenizer = Tokenizer(args['data']['base'])
    args = compute_extra_args(args, tokenizer)

    train_loader, test_loader, oos_loader = get_datasets(args)

    args = compute_nb_steps(args, train_loader)
    model, optimizer, scheduler, args = get_model_opt_scheduler(args, device)

    checkpoint_dir = os.path.join(args['io']['save_path'], 'checkpoints')
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 1 and not args['resume_training']:
        raise ValueError("Save path %s already exists, but not resuming training. Add '--resume_training' to arguments to resume training"%args['io']['save_path'])
    elif args['resume_training']:
        latest_checkpoint = get_last_checkpoint(args['io']['save_path'], map_location)
        model.load_state_dict(latest_checkpoint['model_state_dict'])
        optimizer.load_state_dict(latest_checkpoint['opt_state_dict'])
        scheduler.load_state_dict(latest_checkpoint['scheduler_state_dict'])
    else:
        os.makedirs(args['io']['save_path'])
        os.makedirs(args['io']['save_path'] + 'checkpoints/')
        latest_checkpoint = None

    wandb_init(args)

    if args['verbose']:
        print('Running training for %d steps, %d warmup'%(args['scheduler']['nb_steps'], args['scheduler']['n_warmup_steps']))
        print('Model args: ')
        pprint.pprint(args['model_args'])

    run_training(model, optimizer, scheduler, tokenizer, train_loader, test_loader, oos_loader, device, args, latest_checkpoint)

    if args['multi_gpu']:
        dist.barrier()

    best_checkpoint = get_best_checkpoint(args['io']['save_path'], map_location=map_location)['model_state_dict']
    model.load_state_dict(best_checkpoint)
    compute_factorization_metrics(model, tokenizer, device, args, args['data']['test_path'], 
                save_suffix='test')
    compute_factorization_metrics(model, tokenizer, device, args, args['data']['oos_path'], 
                save_suffix = 'oos')

    if args['multi_gpu']:
        cleanup()

if __name__ == "__main__":
    """
    Test to make sure old version (no relative pe, no repeated pe) make sure results loook same
    TODO: (no particular order)
        * THEY DON'T GET POSITIONAL ENCODINGS OTHERWISE, AND THEY'RE VERY IMPORTANT!!!!
    * DOUBLE CHECK TO MAKE SURE THAT RELATIVE POSITIONAL ENCODINGS ARE WORKING PROPERLY!!!
    * Write some documentation on how the arguments work
        * also cleanup docuemntation that's on github, becasue it's out of date
    * Model seems to be very sensitive to changes in initializations:
        * MultiHeadedAttention.out_proj
        * Initializing with xavier uniform (weights) and 0 (bias) reduces accuracy (on small problem) by 10-12%....
    * How do I add a json to the summary for WANDB?
    * Multi Beam Metrics (i.e. n_beams in args, produce metrics file for each beam size)
    * Different types of positional embeddings:
        * Rotary Embeddings
        * Alibi
        * Concatenating positional embeddings, not adding them
    
    * Different initializations:
        * embeddings
        * WEIGHTS IN ATTENTION SEEM IMPORTANT!!!

    * Shared weights between layers in transformer

    * See about using WANDB sweeps
    * Come up with a small model/dataset that can be trained in ~5 min or something 

    * Is there any benefit to using lightning?
        * I think for multi gpu support it might be useful, but everyhting else I've/would like to do myself
    * Set a seed for model training (what's the best way to handle resuming training? If we resume from training on 
            many numbers (i.e. less than 1 epoch), we don't want to start from the beginning)
    * Remove multi gpu spport in train script - will eventually hopefully move to lightning for this
    """
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume_training', action='store_true', default=False)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)
        config_args['resume_training'] = args.resume_training
    
    
    if torch.cuda.device_count() > 1 and config_args['multi_gpu']:
        run(main, torch.cuda.device_count(), config_args)
    else:
        main(None, config_args)


