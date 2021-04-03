import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from generation_utils import factor
from utils import load_json, save_json
from optimization_utils import test_for_epoch
from data_utils import prepare_dataloader
import torch.autograd.profiler as profiler

# def test_for_epoch(model, loader, tokenizer, loss_func, device):
def load_data_file(args):
    if args['data']['data_loc'].endswith('.json'):
        return load_json(args['data']['data_loc'])
    else:
        return load_json(args['data']['data_loc'] + '2^%d.json'%args['data']['max_pow'])

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
    save_suffix = '_%s'%args['metrics']['save_suffix'] if len(args['metrics']['save_suffix']) > 0 else ''
    metric_df.to_csv(args['io']['save_path'] + 'metric_df%s.csv'%save_suffix)
    metric_df.to_pickle(args['io']['save_path'] + 'metric_df%s.pkl'%save_suffix)
    metrics = compute_metrics(metric_df)
    # Add test loss to metrics
    loader = prepare_dataloader(test_data, args, **args['loader']['test'])
    metrics['test_loss'] = test_for_epoch(model, loader, tokenizer, nn.CrossEntropyLoss(), device)

    metrics['meta'] = {'n_beams' : args['metrics']['n_beams']}
    save_json(metrics, args['io']['save_path'] + 'metrics%s.json'%save_suffix)


def form_factor_df(model, tokenizer, device, items, input_padding, max_encode_size, max_decode_size, n_beams=1, max_num=-1):
    rows = []    
    pbar = tqdm(total = min(len(items), max_num) if max_num > 0 else len(items), leave=False)
    for i, num in enumerate(items):
        if max_num > 0 and i >= max_num:
            break
        if num < 2: continue
        rows.append(factor(num, model, tokenizer, device, input_padding, max_encode_size, max_decode_size, n_beams, return_type='dict'))
        pbar.update(1)
    pbar.update(2)
    pbar.close()
    rows = sum(rows, [])
    df = pd.DataFrame.from_dict(rows)
    return df


def compute_metrics(metric_df):
    metrics = {}
    grouped_by_num = metric_df.groupby('target_num')
    metrics['correct'] = grouped_by_num.agg({
        'correct_product' : 'any',
        'correct_factorization' : 'any'
    }).mean(axis=0).to_dict()
    
    metrics['beam_accuracy'] = metric_df.groupby('beam_idx').agg({
        'correct_product' : 'mean',
        'correct_factorization' : 'mean'
    }).astype(float).to_dict()
    
    metric_df['log_prob_decile'] = pd.qcut(metric_df['log_prob'], q=10).apply(str)
    metrics['by_prob'] = metric_df.groupby('log_prob_decile').agg({
        'correct_product': 'mean',
        'correct_factorization' : 'mean',
        'seq_dist_binary' : 'mean',
        'seq_dist_lev' : 'mean',
        'percent_prime_factors_pred' : 'mean',
    }).astype(float).to_dict()
    
    # Things about the target, we want to take the first one of b/c we're gruping by it. Otherwise, we want all of them as a list
    grouped_by_num = grouped_by_num.agg({k: 'first' if 'target' in k else list for k in list(metric_df) if not k=='target_num'}).reset_index()
    mean_size_product_factorization = ['correct_product_mean', 'correct_product_size', 'correct_factorization_mean', 'correct_factorization_size']
    
    metrics['by_n_target_factors'] = grouped_by_num.groupby('n_target_factors').agg({
        'correct_product' : [lambda x: pd.Series([np.mean([any(y) for y in x])]), 'size'],
        'correct_factorization' : [lambda x: pd.Series([np.mean([any(y) for y in x])]), 'size']
    })
    metrics['by_n_target_factors'].columns = mean_size_product_factorization
    metrics['by_n_target_factors'] = metrics['by_n_target_factors'].to_dict()

    grouped_by_num['number_decile'] = pd.qcut(grouped_by_num['target_num'], q=10).apply(str)
    metrics['by_target_number'] = grouped_by_num.groupby('number_decile').agg({
        'correct_product' : [lambda x: pd.Series([np.mean([any(y) for y in x])]), 'size'],
        'correct_factorization' : [lambda x: pd.Series([np.mean([any(y) for y in x])]), 'size']
    })
    metrics['by_target_number'].columns = mean_size_product_factorization
    metrics['by_target_number'] = metrics['by_target_number'].to_dict()

    grouped_by_num['pct_target_ones_decile'] = pd.qcut(grouped_by_num['pct_target_ones'], q=10).apply(str)
    metrics['by_pct_target_ones'] = grouped_by_num.groupby('pct_target_ones_decile').agg({
        'correct_product' : [lambda x: pd.Series([np.mean([any(y) for y in x])]), 'size'],
        'correct_factorization' : [lambda x: pd.Series([np.mean([any(y) for y in x])]), 'size']
    })
    metrics['by_pct_target_ones'].columns = mean_size_product_factorization
    metrics['by_pct_target_ones'] = metrics['by_pct_target_ones'].to_dict()
    
    return metrics