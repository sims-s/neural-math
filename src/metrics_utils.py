import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from generation_utils import factor
from utils import load_json, save_json, load_data_file
from optimization_utils import test_on_loader
from data_utils import prepare_dataloader
import torch.autograd.profiler as profiler


def compute_factorization_metrics(model, tokenizer, device, args):
    if args['verbose']:
        print('Computing metrics...')
    
    #TODO should pass argument for path of data instead of just taking test...
    numbers = np.load(args['data']['test_path'], mmap_mode='r')
    if args['verbose']:
        print('Factoring...')
    factor_df = form_factor_df(model, tokenizer, device, args['data']['base'], numbers, args['model_args']['max_decode_size'],
                               args['metrics']['n_beams'], args['metrics']['temperature'], args['metrics']['max_num'])
    save_suffix = '_%s'%args['metrics']['save_suffix'] if len(args['metrics']['save_suffix']) > 0 else ''
    factor_df.to_csv(args['io']['save_path'] + 'factor_df%s.csv'%save_suffix, index=False)
    factor_df.to_pickle(args['io']['save_path'] + 'factor_df%s.pkl'%save_suffix)
    metrics = compute_metrics(factor_df)
    # Add test loss to metrics
    loader = prepare_dataloader(args['data']['test_path'], args, **args['loader']['test'])
    metrics['test_loss'] = test_on_loader(model, loader, tokenizer, nn.CrossEntropyLoss(), device, args['optimizer']['gradient_accumulation_steps'])

    metrics['meta'] = {'n_beams' : args['metrics']['n_beams'], 'temperature' : args['metrics']['temperature']}
    save_json(metrics, args['io']['save_path'] + 'metrics%s.json'%save_suffix)


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


def compute_metrics(factor_df):
    metrics = {}
    grouped_by_num = factor_df.groupby('target_num')
    metrics['correct'] = grouped_by_num.agg({
        'correct_product' : 'any',
        'correct_factorization' : 'any'
    }).mean(axis=0).to_dict()
    
    metrics['beam_accuracy'] = factor_df.groupby('beam_idx').agg({
        'correct_product' : 'mean',
        'correct_factorization' : 'mean'
    }).astype(float).to_dict()
    
    factor_df['log_prob_decile'] = pd.qcut(factor_df['log_prob'], q=10).apply(str)
    metrics['by_prob'] = factor_df.groupby('log_prob_decile').agg({
        'correct_product': 'mean',
        'correct_factorization' : 'mean',
        'percent_prime_factors_pred' : 'mean',
    }).astype(float).to_dict()
    
    # Things about the target, we want to take the first one of b/c we're gruping by it. Otherwise, we want all of them as a list
    grouped_by_num = grouped_by_num.agg({k: 'first' if ('target' in k or 'input' in k) else list for k in list(factor_df) if not k=='target_num'}).reset_index()
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

    metrics['pred_same_as_target_beam_0'] = grouped_by_num.groupby(['target_is_prime', 'pred_same_as_target']).agg({
        'correct_product' : [lambda x: pd.Series([np.mean([y[0] for y in x])]), 'size'],
        'correct_factorization' : [lambda x: pd.Series([np.mean([y[0] for y in x])]), 'size']
    })
    metrics['pred_same_as_target_beam_0'].columns = mean_size_product_factorization
    metrics['pred_same_as_target_beam_0'] = metrics['pred_same_as_target_beam_0'].reset_index().to_dict(orient='index')

    nonprime_df = grouped_by_num[~grouped_by_num['target_is_prime']]
    nonprime_df['min_factor_decile'] = pd.qcut(nonprime_df['min_target_prime_factor_if_composite'], q=10, duplicates='drop').apply(str)
    metrics['by_min_factor'] = nonprime_df.groupby('min_factor_decile').agg({
        'correct_product' : [lambda x: pd.Series([np.mean([any(y) for y in x])]), 'size'],
        'correct_factorization' : [lambda x: pd.Series([np.mean([y[0] for y in x])]), 'size']
    })
    metrics['by_min_factor'].columns = mean_size_product_factorization
    metrics['by_min_factor'] = metrics['by_min_factor'].reset_index().to_dict(orient='index')
    
    return metrics