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

    def get_dataset(self, path):
        return EvaluationDataset(path, self.args['data']['base'], form_input=self.form_input, form_label=self.form_label)
    
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

        if not all([c.isdigit() for c in tokenized]) or not len(tokenized) or (len(tokenized)==1 and tokenized[0] == '-'):
            if len(tokenized) and tokenized[0] == '-' and all([c.isdigit() for c in tokenized[1:]]):
                tokenized = tokenized[1:]
                sign = -1
            else:
                return np.nan
        try:
            return sign * base2dec([int(t) for t in tokenized], self.args['data']['base'])
        except (IndexError, ValueError) as e:
            print('huhhh??')
            print(tokenized)
            raise e


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




        raise NotImplementedError('TODO')
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
        nonprime_df['min_factor_decile'] = pd.qcut(nonprime_df['min_target_prime_factor_if_composite'], q=10, duplicates='drop').apply(str)
        metrics['by_min_factor_composite_only'] = nonprime_df.groupby('min_factor_decile').agg({
            'correct_product' : ['size', lambda x: pd.Series([np.mean([any(y) for y in x])])],
            'correct_factorization' : lambda x: pd.Series([np.mean([y[0] for y in x])])
        })
        metrics['by_min_factor_composite_only'].columns = mean_size_product_factorization
        metrics['by_min_factor_composite_only'] = metrics['by_min_factor_composite_only'].reset_index().to_dict(orient='index')
        
        return metrics



class EvaluationDataset(BaseDataset):    
    def __getitem__(self, i):
        expression = self.data[i]
        return {
            'input' : self.form_input(expression, self.base),
            'label' : self.form_label(expression, self.base)
        }

    
