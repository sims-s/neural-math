from problem import Problem
import numpy as np
import sys
sys.path.append('./src/')
import tokenizer
from data_utils import dec2base, base2dec
from problem import Problem, BaseDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class PairwiseAddition(Problem):
    def get_tokenizer(self):
        if self.tokenizer:
            return self.tokenizer
        addition_tokens = [
                '+', # addition
                '[PAD]', #padding,
                '[EOS]', # end of sequence
                '[SOS]', # start of sequence
        ]
        self.special_tokens = addition_tokens
        self.tokenizer = tokenizer.Tokenizer(self.args['data']['base'], self.special_tokens)
        self.update_tokenizer_args(self.tokenizer)
        return self.tokenizer

    def get_dataset(self, path, **dataset_kwargs):
        return AdditionDataset(path, self.args['data']['base'], self.form_input, self.form_label, **dataset_kwargs)

    def form_input(self, n1, n2, base):
        return ['[SOS]'] + dec2base(n1, base) + ['+'] + dec2base(n2, base) + ['[EOS]']

    def form_label(self, n1, n2, base):
        return ['[SOS]'] + dec2base(n1 + n2, base) + ['[EOS]']

    def decode_beam_addition(self, beam, tokenizer, base):
        sos = tokenizer.sos_id
        eos = tokenizer.eos_id
        pad = tokenizer.pad_id
        # Ignore all padding tokens
        beam = [tok for tok in beam if not tok==pad]
        start = None if not sos in beam else beam.index(sos)
        end = None if not eos in beam else beam.index(eos)
        if start is not None and end is not None and (end-start > 1):
            beam = beam[start+1:end]
            try:
                return base2dec(beam, base)
            except (ValueError, TypeError):
                return None
        else:
            return None

    def get_n1_n2(self, input):
        plus_idx = input.index('+')
        n1 = base2dec(input[1:plus_idx], self.args['data']['base'])
        n2 = base2dec(input[plus_idx+1:-1], self.args['data']['base']) 
        return n1, n2

    def postprocess(self, output, logprob, beam_idx, input, **kwargs):
        number_predicted = self.decode_beam_addition(output, self.tokenizer, self.args['data']['base'])
        base_10_n1, base_10_n2 = self.get_n1_n2(input)
        label = self.form_label(base_10_n1, base_10_n2, self.args['data']['base'])
        add_dict = {
            'n1' : base_10_n1, 
            'n2' : base_10_n2, 
            'n1 + n2' : base_10_n1 + base_10_n2, 
            'input_list': input,
            'label_list' : label,
            'input_str' : ''.join([str(c) for c in input ]),
            'label_str' : ''.join([str(c) for c in label]),
            'pred_tokens' : output.tolist(),
            'pred_num' : number_predicted,
            'log_prob' : logprob.item(),
            'beam_idx' : beam_idx,  
        }
        if add_dict['pred_num'] is None:
            add_dict['pred_is_right'] = False
        else:
            add_dict['pred_is_right'] = int(base_10_n1 + base_10_n2) == int(add_dict['pred_num'])
        return add_dict

    def addition_accuracy_plot(self, df, n_beams, save_path=None):
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
        cbar = plt.colorbar(extend='min', label='first correct beam idx')
        # print(list(range(n_beams)))
        cbar.set_ticks(list(range(n_beams)))
        cbar.set_ticklabels(list(range(n_beams))[::-1])
        plt.xlabel('n1')
        plt.ylabel('n2')
        if save_path:
            plt.savefig(save_path)
            plt.clf()
        else:
            plt.show()


    def get_index_first_true(self, x):
        if not isinstance(x, list):
            x = x.tolist()
        if not True in x:
            return -1
        return x.index(True)

    def aggregate_metrics(self, addition_df, save_dir, save_suffix):
        metrics = {}
        n_beams = addition_df['beam_idx'].max() + 1
        grouped_by_input_str = addition_df.groupby('input_str')
        metrics['correct_sum'] = grouped_by_input_str.agg({
            "pred_is_right": "any"
        }).mean(axis=0).to_dict()

        aggd_by_input_str = grouped_by_input_str.agg({
            'n1' : 'first',
            'n2' : 'first',
            'pred_is_right' : self.get_index_first_true
        })

        first_right_beam = aggd_by_input_str['pred_is_right'].value_counts().to_dict()
        metrics['first_right_beam'] = first_right_beam

        if save_suffix:
            if not save_suffix.startswith('_'):
                save_suffix = '_%s'%save_suffix

        if save_dir:
            acc_plot_save_path = save_dir + 'accuracy_plot'
            if save_suffix:
                acc_plot_save_path += save_suffix
            acc_plot_save_path += '.png'
        else:
            acc_plot_save_path = None
        
        self.addition_accuracy_plot(aggd_by_input_str, n_beams, acc_plot_save_path)
        return metrics
        



class AdditionDataset(BaseDataset):
    def __getitem__(self, i):
        n1, n2 = self.data[i]
        output =  {
            'input' : self.form_input(n1, n2, self.base),
            'label' : self.form_label(n1, n2, self.base)
        }
        return output


    def __len__(self):
        return len(self.data)