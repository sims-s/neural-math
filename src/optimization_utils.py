import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import yaml
import os
import wandb
from scheduler import get_linear_schedule_with_warmup

def get_scheduler(args, optimizer):
    # linear_schedule_with_warmup
    if args['scheduler']['type']=='linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, args['scheduler']['n_warmup_steps'], args['scheduler']['nb_steps'])
    else:
        raise ValueError('only using linear_schedule_with_warmup right now')
    return scheduler

def run_batch(model, batch, tokenizer, loss_func, device, grad_accum_steps, train=True):
    numbers = torch.tensor(tokenizer.encode(batch['number'])).to(device)
    labels = torch.tensor(tokenizer.encode(batch['label'])).to(device)
    res = model(numbers, labels[:,:-1])
    loss = loss_func(res.view(-1, len(tokenizer)), labels[:,1:].reshape(-1))
    if train:
        loss = loss / grad_accum_steps
        loss.backward()
    return loss.item()
    
def test_on_loader(model, loader, tokenizer, loss_func, device):
    model.eval()
    pbar = tqdm(total = len(loader), leave=False)
    loss_hist = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            loss_hist.append(run_batch(model, batch, tokenizer, loss_func, device, -1, train=False))
            pbar.update(1)
    pbar.close()
    return np.mean(loss_hist)


def run_epoch(model, opt, scheduler, loader, tokenizer, loss_func, device, args, global_step, global_batches, 
            global_loss_hist, test_loader, oos_loader, pbar):
    max_grad_norm = args['optimizer']['max_grad_norm']
    grad_accum_steps = args['optimizer']['gradient_accumulation_steps']
    checkpoint_every = args['io']['checkpoint_every']
    model.train()
    loss_hist = []
    batch_loss = 0
    for i, batch in enumerate(loader):
        loss = run_batch(model, batch, tokenizer, loss_func, device, grad_accum_steps, train=True)
        batch_loss += loss
        
        if not global_batches % grad_accum_steps:
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            scheduler.step()
            pbar.update(1)
            global_step += 1
            loss_hist.append(batch_loss)
            batch_loss = 0
            model.zero_grad()
        global_batches +=1

        if global_step and not global_step % checkpoint_every and not global_batches % checkpoint_every:
            global_loss_hist = checkpoint(model, opt, test_loader, oos_loader, tokenizer, loss_func, device, 
                                            np.mean(loss_hist), args, global_step, global_batches, global_loss_hist, 
                                            scheduler)
            model.train()
            loss_hist = []
        
        if global_step >= args['scheduler']['nb_steps']:
            break

    if global_step==args['scheduler']['nb_steps']:
            global_loss_hist = checkpoint(model, opt, test_loader, oos_loader, tokenizer, loss_func, device, 
                                            np.mean(loss_hist), args, global_step, global_batches, global_loss_hist, 
                                            scheduler, args['io']['evaluate_final'])

    
    return global_step, global_batches, global_loss_hist


def checkpoint(model, optimizer, test_loader, oos_loader, tokenizer, loss_func, device, train_loss, args, global_step, global_batches, loss_hist, scheduler, force_test = False):
    if args['io']['save_path']:
        if not (global_step / args['io']['checkpoint_every']) % args['io']['evaluate_every'] or force_test:
            test_loss = test_on_loader(model, test_loader, tokenizer, loss_func, device)
            oos_loss = test_on_loader(model, oos_loader, tokenizer, loss_func, device)
        else:
            test_loss = np.nan
            oos_loss = np.nan
        if args['verbose']:
            tqdm.write('Train: %.6f, Test: %.6f, OoS: %.6f'%(train_loss, test_loss, oos_loss))

        
        loss_hist.append([global_step, train_loss, test_loss, oos_loss])
        loss_df = pd.DataFrame.from_records(loss_hist, columns = ['step', 'train_loss', 'test_loss', 'oos_loss'])
        loss_df.to_csv(args['io']['save_path'] + 'loss_hist.csv', index=False)

        if args['wandb']['enabled']:
            wandb.log({'train_loss' : train_loss, 'test_loss' : test_loss, 'oos_loss' : oos_loss, 'step' : global_step})

        torch.save({'model_state_dict' : model.state_dict(), 
                    'opt_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'train_loss' : train_loss, 
                    'test_loss' : test_loss,
                    'oos_loss' : oos_loss,
                    'args': args, 
                    'global_step' : global_step,
                    'global_batches' : global_batches,},
                    '%s/%d_%.4f.pt'%(os.path.join(args['io']['save_path'], 'checkpoints'), global_step, test_loss))

    return loss_hist


def run_training(model, optimizer, scheduler, tokenizer, train_loader, test_loader, oos_loader, device, args, latest_checkpoint):
    with open(args['io']['save_path'] + 'config.yaml', 'w') as f:
        yaml.dump(args, f)

    if args['verbose']:
        if latest_checkpoint is None:
            print('Starting training!')
        else:
            print('Resuming Training!')
    
    loss_func = nn.CrossEntropyLoss()
    pbar = tqdm(total = args['scheduler']['nb_steps'], leave=False)
    if args['wandb']['enabled']:
        wandb.watch(model, criterion=loss_func, **args['wandb']['watch_args'])

    if latest_checkpoint is None:
        global_loss_hist = []
        global_step = 0
        global_batches = 0 
    else:
        args = latest_checkpoint['args']
        global_loss_hist = pd.read_csv(os.path.join(args['io']['save_path']) + 'loss_hist.csv').values.tolist()
        global_step = latest_checkpoint['global_step']
        global_batches = latest_checkpoint['global_batches']
        pbar.update(global_step)
    model.zero_grad()
    for i in range(args['scheduler']['nb_epochs']):
        global_step, global_batches, global_loss_hist = run_epoch(model, optimizer, scheduler, train_loader, tokenizer, loss_func, device, args, global_step, 
                                 global_batches, global_loss_hist, test_loader, oos_loader, pbar)
    pbar.close()

    if args['verbose']:
        print('Finished Training!')