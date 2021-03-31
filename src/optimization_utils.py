import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import yaml

def run_batch(model, batch, tokenizer, loss_func, device, backward=True):
    numbers = torch.tensor(tokenizer(batch['number'])).to(device)
    labels = torch.tensor(tokenizer(batch['label'])).to(device)
    res = model(numbers, labels[:,:-1])
    loss = loss_func(res.view(-1, 5), labels[:,1:].reshape(-1))
    if backward:
        loss.backward()
    return loss.item()
    

def train_for_epoch(model, opt, scheduler, loader, tokenizer, loss_func, device):
    model.train()
    pbar = tqdm(total = len(loader), leave=False)
    loss_hist = [] 
    for i, batch in enumerate(loader):
        model.zero_grad()
        loss = run_batch(model, batch, tokenizer, loss_func, device, backward=True)
        opt.step()
        scheduler.step()
        pbar.update(1)
        loss_hist.append(loss)
    pbar.close()
    mean_loss = np.mean(loss_hist)
    
    return mean_loss

def test_for_epoch(model, loader, tokenizer, loss_func, device):
    model.eval()
    pbar = tqdm(total = len(loader), leave=False)
    loss_hist = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            loss_hist.append(run_batch(model, batch, tokenizer, loss_func, device, backward=False))
            pbar.update(1)
    pbar.close()
    return np.mean(loss_hist)


def run_training(model, optimizer, scheduler, tokenizer, train_loader, test_loader, device, args):
    with open(args['io']['save_path'] + 'config.yaml', 'w') as f:
        yaml.dump(args, f)

    if args['verbose']:
        print('Starting training!')
    
    loss_func = nn.CrossEntropyLoss()
    pbar = tqdm(total = args['scheduler']['nb_epochs'], leave=False)
    for i in range(args['scheduler']['nb_epochs']):
        train_loss = train_for_epoch(model, optimizer, scheduler, train_loader, tokenizer, loss_func, device)
        test_loss = test_for_epoch(model, test_loader, tokenizer, loss_func, device)
        if args['io']['save_path']:
            torch.save({'model_state_dict' : model.state_dict(), 
                        'opt_state_dict': optimizer.state_dict(), 
                        'train_loss' : train_loss, 
                        'test_loss' : test_loss,
                        'args': args, 
                        'epoch' : i},
                        '%s/%d_%.4f.pt'%(args['io']['save_path'], i, test_loss))
        if args['verbose']:
            tqdm.write('Train: %.6f, Test: %.6f'%(train_loss, test_loss))
        pbar.update(1)
    pbar.close()

    if args['verbose']:
        print('Finished Training!')