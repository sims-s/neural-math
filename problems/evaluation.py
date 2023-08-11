import numpy as np
import pandas as pd
from sympy import factorint
from problem import Problem
import sys
sys.path.append('./src/')
import utils
import tokenizer
from data_utils import dec2base, base2dec
from problem import Problem, BaseDataset
from sympy.ntheory.primetest import isprime
import warnings
import torch
sys.path.append('./scripts/dataset_generation/')
import evaluation_configs, generate_evaluation_data


class Evaluation(Problem):
    def get_tokenizer(self):
        if self.tokenizer:
            return self.tokenizer
        eval_tokens = [
            '+',
            '—', # subtraction
            '-', # negative number
            '*',
            '(',
            ')',
            '[PAD]',
            '[EOS]',
            '[SOS]',
        ]
        self.special_tokens = eval_tokens
        self.tokenizer = tokenizer.Tokenizer(self.args['data']['base'], self.special_tokens)
        self.update_tokenizer_args(self.tokenizer)
        return self.tokenizer

    def form_label(self, expression, base):
        # if isinstance(factors, int):
            # factors = factorint(factors, multiple=True)
        # factors = [dec2base(f, base) for f in factors]
        # factors = ['[SOS]'] + utils.list_join(factors, '*') + ['[EOS]']
        result = eval(expression.replace('—', '-'))
        return ['[SOS]'] + dec2base(result, base) + ['[EOS]']
        return factors

    def form_input(self, expresssion, base, to_expression=False):
        split_tokens = self.special_tokens[:-3]
        tokens = ['[SOS]'] if not to_expression else []
        char_idx = 0
        start = 0
        while char_idx < len(expresssion)+1:
            if char_idx == len(expresssion):
                if start < len(expresssion):
                    if to_expression:
                        tokens.append(base2dec(expresssion[start:], base))
                    else:
                        tokens.extend(dec2base(int(expresssion[start:]), base))
                break
            if expresssion[char_idx] in split_tokens:
                if char_idx > 0 and not expresssion[char_idx-1] in split_tokens:
                    if to_expression:
                        tokens.append(base2dec(expresssion[start:char_idx], base))
                    else:
                        tokens.extend(dec2base(int(expresssion[start:char_idx]), base))
                    tokens.append(expresssion[char_idx])
                    start = char_idx + 1
                else:
                    tokens.append(expresssion[char_idx])
                    start = char_idx + 1
        
            char_idx += 1
        if not to_expression:
            tokens.append('[EOS]')
        return tokens

    def get_dataset(self, path, **dataset_kwargs):
        if path.endswith('.csv'):
            return EvaluationDataset(path, self.args['data']['base'], form_input=self.form_input, form_label=self.form_label)
        elif path.lower().startswith('streaming_'):
            config_name = "_".join(path.split('_')[1:])
            if not config_name in evaluation_configs.CONFIGS:
                raise ValueError(f'weird streaming_suffix/config: {config_name}')
            return StreamingEvaluationDataset(self.args['data']['base'], self.form_input, self.form_label, config_name, **dataset_kwargs)
        else:
            raise 
    
    def compute_output(self, tokenized):
        if tokenized[:5] == '[SOS]':
            tokenized = tokenized[5:]
        first_eos = tokenized.index('[EOS]') if '[EOS]' in tokenized else len(tokenized)
        tokenized = [tok for tok in tokenized[:first_eos].split(' ') if len(tok)]
        # print(tokenized)
        sign = 1

        if not all([c.isdigit() or (i==0 and c=='-') for i, c in enumerate(tokenized)]) or not len(tokenized):
            return np.nan
        # print(tokenized)
        return base2dec([int(t) if t.isdigit() else t for t in tokenized], self.args['data']['base'])


    def postprocess(self, output, log_prob, beam_idx, input, postprocess_minimal=False, **kwargs):
        expr_tokenized = self.form_input(input[1:-1], self.args['data']['base'], to_expression=True)
        expr = ''.join([str(x) for x in expr_tokenized])
        tokenized = self.tokenizer.decode(output, decode_special=True)
        output_num = self.compute_output(tokenized)
    
        information = {
            'expression' : expr,
            'value' : eval(expr.replace('—', '-')),
            'model_input' : [str(c) for c in input],
            'beam_idx' : beam_idx,
            'log_prob' : log_prob.item(),
            'output_toks' : [t for t in tokenized.split(' ') if len(t)],
            'pred' : output_num
        }
        information['correct_value'] = information['value'] == information['pred']

        return information


    def aggregate_metrics(self, factor_df, save_dir=None, save_suffix=None):
        metrics = {}
        grouped_by_expr = factor_df.groupby('expression')
        metrics.update(grouped_by_expr.agg({
            'correct_value' : 'any'
        }).mean(axis=0).to_dict())

        metrics['beam_accuracy'] = factor_df.groupby('beam_idx').agg({
            'correct_value': 'mean'
        }).astype(float).to_dict()


        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            factor_df['log_prob_decile'] = pd.qcut(factor_df['log_prob'], q=10, duplicates='drop').apply(str)
        metrics['by_prob'] = factor_df.groupby('log_prob_decile').agg({
            'beam_idx' : 'size',
            'correct_value': 'mean',
        }).rename({'beam_idx' : 'group_size'}, axis=1).astype(float).to_dict()
        
        return metrics




class EvaluationDataset(BaseDataset):    
    def __getitem__(self, i):
        expression = self.data[i]
        return {
            'input' : self.form_input(expression, self.base),
            'label' : self.form_label(expression, self.base)
        }


class StreamingEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, base, form_input, form_label, config, size, config_value_key = 'in_sample_vals', cache_size=50_000, force_oos=False, invalid_exprs=None):
        self.base = base
        self.form_input = form_input
        self.form_label = form_label
        config = evaluation_configs.CONFIGS[config]
        for k in ['exp_vs_num_prob', 'functions', 'in_sample_vals', 'oos_vals', 'full_vals']:
            setattr(self, k, config[k])
        self.size = size
        self.sampler_cache = generate_evaluation_data.ExpressionSampleCache(cache_size, self.functions, getattr(self, config_value_key), self.exp_vs_num_prob)
        self.force_oos = force_oos
        if invalid_exprs is None:
            invalid_exprs = set()
        self.invalid_exprs = invalid_exprs

    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        tree = generate_evaluation_data.sample_expression(self.sampler_cache)
        if self.force_oos and not set([node.data['val'] for _, node in tree.nodes.items() if node.data['type']=='num']).intersection(set(self.oos_vals)):
            return self.__getitem__(i)
        expression = generate_evaluation_data.convert_tree_to_expression(tree)
        if expression in self.invalid_exprs:
            return self.__getitem__(i)
        return {
            'input' : self.form_input(expression, self.base),
            'label' : self.form_label(expression, self.base)
        }
        
    


        

    
