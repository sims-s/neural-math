import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from generation_utils import factor, model_add
from utils import load_json, save_json, load_data_file
from optimization_utils import test_on_loader
from data_utils import prepare_dataloader
import torch.autograd.profiler as profiler
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import lru_cache



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


def addition_accuracy_plot(df, n_beams, save_path=None):
    fig = plt.gcf()
    # fig.set_size_inches(12, 12)
    cmap = cm.get_cmap('winter', n_beams)
    cmap.set_under('red')
    # print('what should be on the x axis: ', aggd_by_input_str['n1'].)
    plt.scatter(x=df['n1'], 
                y=df['n2'], 
                c=df['pred_is_right'].apply(lambda x: -1 if x==-1 else n_beams-1-x), 
                alpha=.5, 
                cmap = cmap, 
                vmin=0
        )
    cbar = plt.colorbar(extend='min', label='first correct beam')
    cbar.set_ticks(list(range(n_beams)))
    cbar.set_ticklabels(list(range(n_beams))[::-1])
    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def get_index_first_true(x):
    if not isinstance(x, list):
        x = x.tolist()
    if not True in x:
        return -1
    return x.index(True)

def aggregate_addition_metrics(addition_df, save_path, save_suffix):
    metrics = {}
    n_beams = addition_df['beam_idx'].max()
    grouped_by_input_str = addition_df.groupby('input_str')
    metrics['correct_sum'] = grouped_by_input_str.agg({
        "pred_is_right": "any"
    }).mean(axis=0).to_dict()

    aggd_by_input_str = grouped_by_input_str.agg({
        'n1' : 'first',
        'n2' : 'first',
        'pred_is_right' : get_index_first_true
    })

    first_right_beam = aggd_by_input_str['pred_is_right'].value_counts().to_dict()
    metrics['first_right_beam'] = first_right_beam

    # non_tuple = {}
    # for k, v in first_right_beam.items():
    #     non_tuple[k[0]] = v
    # first_right_beam = non_tuple
    # metrics['first_right_beam'] = first_right_beam

    if save_suffix:
        if not save_suffix.startswith('_'):
            save_suffix = '_%s'%save_suffix

    acc_plot_save_path = save_path + 'accuracy_plot'
    if save_suffix:
        acc_plot_save_path += save_suffix
    acc_plot_save_path += '.png'
    
    addition_accuracy_plot(aggd_by_input_str, n_beams, acc_plot_save_path)


    return metrics

def aggregate_factorization_metrics(factor_df):
    metrics = {}
    grouped_by_num = factor_df.groupby('input_num')
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
        'beam_idx' : 'size',
        'correct_product': 'mean',
        'correct_factorization' : 'mean',
        'percent_prime_factors_pred' : 'mean',
    }).rename({'beam_idx' : 'group_size'}, axis=1).astype(float).to_dict()
    
    # Things about the target, we want to take the first one of b/c we're gruping by it. Otherwise, we want all of them as a list
    grouped_by_num = grouped_by_num.agg({k: 'first' if ('target' in k or 'input' in k) else list for k in list(factor_df) if not k=='input_num'}).reset_index()
    mean_size_product_factorization = ['group_size', 'correct_product_mean', 'correct_factorization_mean']
    
    metrics['by_num_target_factors'] = grouped_by_num.groupby('num_target_factors').agg({
        'correct_product' : ['size', lambda x: pd.Series([np.mean([any(y) for y in x])])],
        'correct_factorization' : lambda x: pd.Series([np.mean([any(y) for y in x])])
    })
    metrics['by_num_target_factors'].columns = mean_size_product_factorization
    metrics['by_num_target_factors'] = metrics['by_num_target_factors'].to_dict()

    grouped_by_num['number_decile'] = pd.qcut(grouped_by_num['input_num'], q=10).apply(str)
    metrics['by_input_num'] = grouped_by_num.groupby('number_decile').agg({
        'correct_product' : ['size', lambda x: pd.Series([np.mean([any(y) for y in x])])],
        'correct_factorization' : lambda x: pd.Series([np.mean([any(y) for y in x])])
    })
    metrics['by_input_num'].columns = mean_size_product_factorization
    metrics['by_input_num'] = metrics['by_input_num'].to_dict()

    metrics['pred_same_as_input_beam_0'] = grouped_by_num.groupby(['input_is_prime', 'pred_same_as_input']).agg({
        'correct_product' : ['size', lambda x: pd.Series([np.mean([y[0] for y in x])])],
        'correct_factorization' : lambda x: pd.Series([np.mean([y[0] for y in x])])
    })
    metrics['pred_same_as_input_beam_0'].columns = mean_size_product_factorization
    metrics['pred_same_as_input_beam_0'] = metrics['pred_same_as_input_beam_0'].reset_index().to_dict(orient='index')

    nonprime_df = grouped_by_num[~grouped_by_num['input_is_prime']]
    nonprime_df['min_factor_decile'] = pd.qcut(nonprime_df['min_target_prime_factor_if_composite'], q=10, duplicates='drop').apply(str)
    metrics['by_min_factor_composite_only'] = nonprime_df.groupby('min_factor_decile').agg({
        'correct_product' : ['size', lambda x: pd.Series([np.mean([any(y) for y in x])])],
        'correct_factorization' : lambda x: pd.Series([np.mean([y[0] for y in x])])
    })
    metrics['by_min_factor_composite_only'].columns = mean_size_product_factorization
    metrics['by_min_factor_composite_only'] = metrics['by_min_factor_composite_only'].reset_index().to_dict(orient='index')
    
    return metrics