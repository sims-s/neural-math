import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import yaml
import os
import scheduler as scheduler_mod
import sys
sys.path.append('src/')
import data_utils
import warnings
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from contextlib import nullcontext

# def trace_handler(out_dir, p):
#     output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
#     print(output)
#     # p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
#     p.export_chrome_trace(f"{out_dir}/trace_" + str(p.step_num) + ".json")

# class NullContext(nullcontext):
#     def step(self):
#         pass



def run_batch(model, batch, tokenizer, scaler, loss_func, device, grad_accum_steps, train=True, dataset=None):
    numbers = torch.tensor(tokenizer.encode(batch['input'])).to(device)
    labels = torch.tensor(tokenizer.encode(batch['label'])).to(device)

    amp_context = nullcontext() if not scaler else torch.autocast('cuda', dtype=torch.float16)
    with amp_context:
        res = model(numbers, labels[:,:-1])
        loss = loss_func(res.view(-1, len(tokenizer)), labels[:,1:].reshape(-1))
    if train:
        loss = loss.mean()
        loss = loss / grad_accum_steps
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    else:
        if dataset and isinstance(dataset, data_utils.FactorizationDatasetPropLoss):
            dataset.update_weights(loss)
        loss = loss.mean()
    
    return loss

    
def test_on_loader(model, loader, tokenizer, loss_func, device):
    model.eval()
    pbar = tqdm(total = len(loader), leave=False)
    loss_hist = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            loss_hist.append(run_batch(model, batch, tokenizer, None, loss_func, device, -1, train=False).data.cpu().numpy().item())
            pbar.update(1)
    pbar.close()
    return np.mean(loss_hist)


def run_epoch(model, opt, scheduler, loader, tokenizer, scaler, loss_func, device, args, global_step, global_batches, 
            global_loss_hist, test_loader, oos_loader, pbar):
    max_grad_norm = args['optimizer']['max_grad_norm']
    grad_accum_steps = args['optimizer']['gradient_accumulation_steps']
    checkpoint_every = args['io']['checkpoint_every']
    model.train()
    loss_hist = []
    batch_loss = 0

    # run_profiling = True
    # if run_profiling:
    #     profiler_dir = args['io']['save_path'] + 'profiler/'
    #     os.makedirs(profiler_dir, exist_ok=True)
    #     profiler_context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #                   schedule = torch.profiler.schedule(wait=1, warmup=2, active=4), on_trace_ready=lambda p: trace_handler(profiler_dir, p),
    #                   record_shapes=True, profile_memory=True)
    # else:
    #     profiler_context = NullContext()

    # with profiler_context as p:
    for i, batch in enumerate(loader):
        loss = run_batch(model, batch, tokenizer, scaler, loss_func, device, grad_accum_steps, train=True)
        batch_loss += loss
        global_batches +=1
        
        if grad_accum_steps <=1 or not global_batches % grad_accum_steps:
            if max_grad_norm > 0:
                if scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            # p.step()
            if isinstance(scheduler, scheduler_mod.ReduceLROnPlateauWithWarmup):
                scheduler.step(batch_loss)
            else:
                scheduler.step()
            pbar.update(1)
            global_step += 1
            loss_hist.append(batch_loss.item())
            batch_loss = 0
            zero_grad(model)

        if global_step and not global_step % checkpoint_every and not global_batches % checkpoint_every:
            global_loss_hist = checkpoint(model, opt, test_loader, oos_loader, tokenizer, scaler, loss_func, device, 
                                            np.mean(loss_hist), args, global_step, global_batches, global_loss_hist, 
                                            scheduler)
            model.train()
            loss_hist = []
        
        if global_step >= args['scheduler']['scheduler_args']['nb_steps']:
            break
    if global_step==args['scheduler']['scheduler_args']['nb_steps']:
            global_loss_hist = checkpoint(model, opt, test_loader, oos_loader, tokenizer, scaler, loss_func, device, 
                                            np.mean(loss_hist), args, global_step, global_batches, global_loss_hist, 
                                            scheduler, args['io']['evaluate_final'])

    
    return global_step, global_batches, global_loss_hist


def checkpoint(model, optimizer, test_loader, oos_loader, tokenizer, scaler, loss_func, device, train_loss, args, global_step, global_batches, loss_hist, scheduler, force_test = False):
    if args['io']['save_path']:
        if not (global_step / args['io']['checkpoint_every']) % args['io']['evaluate_every'] or force_test:
            test_loss = test_on_loader(model, test_loader, tokenizer, loss_func, device)
            oos_loss = test_on_loader(model, oos_loader, tokenizer, loss_func, device)
        else:
            test_loss = np.nan
            oos_loss = np.nan
        if args['verbose']:
            tqdm.write('Train: %.9f, Test: %.9f, OoS: %.9f'%(train_loss, test_loss, oos_loss))
        # for i, param_group in enumerate(optimizer.param_groups):
        #     print(param_group['lr'])

        
        loss_hist.append([global_step, train_loss, test_loss, oos_loss])
        loss_df = pd.DataFrame.from_records(loss_hist, columns = ['step', 'train_loss', 'test_loss', 'oos_loss'])
        loss_df.to_csv(args['io']['save_path'] + 'loss_hist.csv', index=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save({'model_state_dict' : model.state_dict(), 
                        'opt_state_dict': optimizer.state_dict(), 
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'scaler_state_dict' : scaler.state_dict() if scaler is not None else None,
                        'train_loss' : train_loss, 
                        'test_loss' : test_loss,
                        'oos_loss' : oos_loss,
                        'args': args, 
                        'global_step' : global_step,
                        'global_batches' : global_batches,},
                        '%s/%d_%.4f.pt'%(os.path.join(args['io']['save_path'], 'checkpoints'), global_step, test_loss))

    return loss_hist

def zero_grad(model):
    for p in model.parameters():
        p.grad = None

def run_training(model, optimizer, scheduler, tokenizer, train_loader, test_loader, oos_loader, device, args, latest_checkpoint):
    with open(args['io']['save_path'] + 'config.yaml', 'w') as f:
        yaml.dump(args, f)

    if args['verbose']:
        if latest_checkpoint is None:
            print('Starting training!')
        else:
            print('Resuming Training!')
    
    loss_func = nn.CrossEntropyLoss(reduction='none')
    pbar = tqdm(total = args['scheduler']['scheduler_args']['nb_steps'], leave=False)

    mixed_precision = args.get('mixed_precision', True)
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if latest_checkpoint is None:
        global_loss_hist = []
        global_step = 0
        global_batches = 0 
    else:
        args = latest_checkpoint['args']
        global_loss_hist = pd.read_csv(os.path.join(args['io']['save_path']) + 'loss_hist.csv').values.tolist()
        global_step = latest_checkpoint['global_step']
        global_batches = latest_checkpoint['global_batches']
        if scaler and latest_checkpoint['scaler_state_dict']:
            scaler.load_state_dict(latest_checkpoint['scaler_state_dict'])
        pbar.update(global_step)
    zero_grad(model)
    for i in range(args['scheduler']['nb_epochs']):
        global_step, global_batches, global_loss_hist = run_epoch(model, optimizer, scheduler, train_loader, tokenizer, scaler, loss_func, device, args, global_step, 
                                 global_batches, global_loss_hist, test_loader, oos_loader, pbar)
    pbar.close()

    if args['verbose']:
        print('Finished Training!')