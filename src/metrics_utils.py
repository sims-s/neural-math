import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from utils import load_json, save_json, load_data_file
from optimization_utils import test_on_loader




def compute_metrics_any_model(model, tokenizer, device, args, data_path=None, save_suffix=''):
    if args['model_type']=='addition':
        compute_addition_metrics(model, tokenizer, device, args, data_path=data_path, save_suffix=save_suffix)
    elif args['model_type']=='factorization':
        compute_factorization_metrics(model, tokenizer, device, args, data_path = data_path, save_suffix = save_suffix)    

def compute_addition_metrics(model, tokenizer, device, args, data_path=None, save_suffix=''):
    if not data_path:
        data_path = args['data']['test_path']
    numbers = np.load(data_path, mmap_mode='r')
    # print('='*100)
    # print('DAta path: ', data_path)
    # print('numbers sahpe: ', numbers.shape)
    # print(numbers[:,0][5])
    # print(numbers[:,0][1231])
    # print(numbers[:,0][664])
    # print(numbers[:,0][-1])
    # print('='*100)
    if args['verbose']:
        print('read from data path %s'%data_path)
        print('adding...')
    addition_df = form_addition_df(model, tokenizer, device, args['data']['base'], numbers, args['model_args']['max_decode_size'],
                    args['metrics']['n_beams'], args['metrics']['temperature'], args['metrics']['max_num'])


    if save_suffix:
        save_suffix = '_%s'%save_suffix
    else:
        save_suffix = '_%s'%args['metrics']['save_suffix'] if len(args['metrics']['save_suffix']) > 0 else ''
    addition_df.to_csv(args['io']['save_path'] + 'addition_df%s.csv'%save_suffix, index=False)

    metrics = aggregate_addition_metrics(addition_df, args['io']['save_path'], save_suffix)
    save_json(metrics, args['io']['save_path'] + f'metrics{save_suffix}.json')

def compute_factorization_metrics(model, tokenizer, device, args, data_path = None, save_suffix = ''):
    if args['verbose']:
        print('Computing metrics...')
    
    if not data_path:
        data_path = args['data']['test_path']
    numbers = np.load(data_path, mmap_mode='r')

    if args['verbose']:
        print('read from data path %s'%data_path)
        print('Factoring...')
    factor_df = form_factor_df(model, tokenizer, device, args['data']['base'], numbers, args['model_args']['max_decode_size'],
                               args['metrics']['n_beams'], args['metrics']['temperature'], args['metrics']['max_num'])
    if save_suffix:
        save_suffix = '_%s'%save_suffix
    else:
        save_suffix = '_%s'%args['metrics']['save_suffix'] if len(args['metrics']['save_suffix']) > 0 else ''
    factor_df.to_csv(args['io']['save_path'] + 'factor_df%s.csv'%save_suffix, index=False)
    # factor_df.to_pickle(args['io']['save_path'] + 'factor_df%s.pkl'%save_suffix)
    metrics = aggregate_factorization_metrics(factor_df)
    # Add test loss to metrics
    loader_args = args['loader']['oos'] if data_path == args['data']['oos_path'] else args['loader']['test']
    loader = prepare_dataloader(data_path, args, **loader_args)
    metrics['loss'] = test_on_loader(model, loader, tokenizer, nn.CrossEntropyLoss(), device)
    metrics['meta'] = {'n_beams' : args['metrics']['n_beams'], 'temperature' : args['metrics']['temperature']}
    save_json(metrics, args['io']['save_path'] + 'metrics%s.json'%save_suffix)
    # if args['wandb']['enabled']:
    #     wandb.run.summary['metrics'] = metrics

def form_addition_df(model, tokenizer, device, base, number_pairs, max_decode_size, n_beams=1, temperature=1.0, max_num=-1, sample_pct=1):
    rows = []
    pbar = tqdm(total = min(len(number_pairs), max_num) if max_num >=0 else len(number_pairs), leave=False)
    for i, (n1, n2) in enumerate(number_pairs):
        if max_num >=0 and i>=max_num:
            break
        if np.random.rand() < 1 - sample_pct:
            pbar.update(1)
            continue
        rows.extend(model_add(n1, n2, base, model, tokenizer, device, max_decode_size, n_beams, temperature, return_type='dict'))
        pbar.update(1)
    df = pd.DataFrame.from_dict(rows)
    return df

def form_factor_df(model, tokenizer, device, base, numbers, max_decode_size, n_beams=1, temperature=1.0, max_num=-1, postprocess_minimal=False):
    rows = []    
    pbar = tqdm(total = min(len(numbers), max_num) if max_num > 0 else len(numbers), leave=False)
    for i in range(numbers.shape[0]):
        num = numbers[i]
        if max_num > 0 and i >= max_num:
            break
        if num < 2: continue
        rows.append(factor(num, base, model, tokenizer, device, max_decode_size, n_beams, temperature, return_type='dict', postprocess_minimal=postprocess_minimal))
        pbar.update(1)
    pbar.update(2)
    pbar.close()
    rows = sum(rows, [])
    df = pd.DataFrame.from_dict(rows)
    return df


