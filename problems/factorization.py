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


class Factorization(Problem):
    def get_tokenizer(self):
        if self.tokenizer:
            return self.tokenizer
        factorization_tokens = [
                '*', # multiplication
                '[PAD]', #padding,
                '[EOS]', # end of sequence
                '[SOS]', # start of sequence
        ]
        self.special_tokens = factorization_tokens
        self.tokenizer = tokenizer.Tokenizer(self.args['data']['base'], self.special_tokens)
        self.update_tokenizer_args(self.tokenizer)
        return self.tokenizer

    def form_label(self, factors, base):
        if isinstance(factors, int):
            factors = factorint(factors, multiple=True)
        factors = [dec2base(f, base) for f in factors]
        factors = ['[SOS]'] + utils.list_join(factors, '*') + ['[EOS]']
        return factors

    def form_input(self, number, base):
        number = dec2base(number, base)
        return ['[SOS]'] + number + ['[EOS]']

    def get_dataset(self, path):
        return FactorizationDataset(path, self.args['data']['base'], form_input=self.form_input, form_label=self.form_label)
    
    def compute_base10_from_input(self, input):
        if len(input) <=1:
            return None
        return base2dec(input[1:-1], self.args['data']['base'])

    def extract_factors(self, factor_list):
        chunked = []
        chunk_start_idx = 0
        for i, token in enumerate(factor_list):
            if token=='*' or token =='[EOS]':
                chunked.append(factor_list[chunk_start_idx:i])
                chunk_start_idx = i + 1
                if token == "[EOS]":
                    break
            elif i==len(factor_list)-1:
                chunked.append(factor_list[chunk_start_idx:i+1])
        return chunked

    def postprocess(self, output, log_prob, beam_idx, input, postprocess_minimal=False, **kwargs):
        base_10_number = self.compute_base10_from_input(input)
        information = {
            'input_num' : base_10_number,
            'model_input' : [str(c) for c in input],
            'beam_idx' : beam_idx,
            'log_prob' : log_prob.item(),
        }

        tokenized = self.tokenizer.decode(output, decode_special=True)
        factor_list = utils.drop_from_iterable(tokenized.split(' '), ['[SOS]', '[PAD]', '[EOS]'])
        factor_list = self.extract_factors(factor_list)

        try:
            factors = [base2dec([int(digit) for digit in num], self.args['data']['base']) for num in factor_list]
        except (IndexError, ValueError):
            factors = []
            
        information['pred_str'] = tokenized
        information['pred_factor_list'] = factors

        if len(information['pred_factor_list']) > 0:
            information['product'] = np.prod(factors)
        else:
            information['product'] = np.nan

        information['num_pred_factors'] = len(information['pred_factor_list'])
        information['num_prime_factors_pred'] = np.sum([isprime(f) for f in factors])
        information['percent_prime_factors_pred'] = information['num_prime_factors_pred'] / information['num_pred_factors']

        information['correct_product'] = information['product']==base_10_number
        information['correct_factorization'] = information['correct_product'] & all([isprime(n) for n in information['pred_factor_list']])
        information['pred_same_as_input'] = ' '.join(information['model_input'])==information['pred_str'].replace('_', '').strip()


        if not postprocess_minimal:
            information['input_is_prime'] = isprime(base_10_number)
            
            # TODO: data_utils.form_label calls factorint, and so does computation of target_factor_list. Should remove redundant computation
            information['target_factor_str'] = ' '.join([str(c) for c in self.form_label(base_10_number, self.args['data']['base'])])
            information['target_factor_list'] = sum([[k]*v for k, v in factorint(base_10_number).items()], [])

            information['num_target_factors'] = len(information['target_factor_list'])

            information['min_target_prime_factor_if_composite'] = -1 if len(information['target_factor_list'])==1 else min(information['target_factor_list'])

        return information


    def aggregate_metrics(self, factor_df, save_dir=None, save_suffix=None):
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
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            factor_df['log_prob_decile'] = pd.qcut(factor_df['log_prob'], q=10, duplicates='drop').apply(str)
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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nonprime_df['min_factor_decile'] = pd.qcut(nonprime_df['min_target_prime_factor_if_composite'], q=10, duplicates='drop').apply(str)
        metrics['by_min_factor_composite_only'] = nonprime_df.groupby('min_factor_decile').agg({
            'correct_product' : ['size', lambda x: pd.Series([np.mean([any(y) for y in x])])],
            'correct_factorization' : lambda x: pd.Series([np.mean([y[0] for y in x])])
        })
        metrics['by_min_factor_composite_only'].columns = mean_size_product_factorization
        metrics['by_min_factor_composite_only'] = metrics['by_min_factor_composite_only'].reset_index().to_dict(orient='index')
        
        return metrics



class FactorizationDataset(BaseDataset):    
    def __getitem__(self, i):
        factors = self.data[i]
        factors = factors[factors > 1]
        return {
            'input' : self.form_input(factors.prod(), self.base),
            'label' : self.form_label(factors, self.base)
        }

    
