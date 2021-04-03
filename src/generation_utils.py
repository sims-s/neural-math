import numpy as np
import torch
import pandas as pd
from data_utils import dec2bin, bin2dec, pad_input, FactorizationDataset, gfm
from Levenshtein import distance as levenshtein_distance

# This function could be optimized a bit. For convineint I create all possible sequences @ each step w/ repeat_interlave, which is n_beams*len(tokenizer) sequences
# becaue it makes indexing convinient.
# According to the pytorch memory profiler, this takes about 1/6 of the running time of the total function. 
def decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams):
    sequences = torch.tensor(tokenizer('1')).to(device).unsqueeze(0)
    sequence_log_probs = torch.tensor([[0]]).to(device)
    eos_token = tokenizer('.')[0]
    
    for i in range(max_decode_size-1):
        with torch.no_grad():
            output = model.decode(sequences, 
                                  memory.repeat(1,sequences.size(0),1), 
                                  memory_key_padding_mask.repeat(sequences.size(0), 1))[:,-1]
            output = torch.log_softmax(output, dim=-1)
        
        # Compue all possible next sequences
        sequences = torch.repeat_interleave(sequences, len(tokenizer), 0)
        next_tokens = torch.arange(len(tokenizer)).repeat(output.size(0)).unsqueeze(1).to(sequences.device)
        next_sequences = torch.cat((sequences, next_tokens), dim=1)
        
        # Get the indiclies of the highest probability sequences
        next_token_log_probs = output.view(-1,1)
        # For cases when the model migth predict a non padding token after an end of sequence token
        # Manually set the log probability to be very low so it's never chosen to be decdoed
        # Additionally, break if all sequences have eos tokens
        seq_has_eos = torch.argmax((sequences==tokenizer('.')[0]).int(), dim=1).view(-1,1) > 0
        if torch.all(seq_has_eos):
            # undo repeat_interleave
            sequences = sequences[::len(tokenizer)]
            break

        # Compute all possible next log probabilities
        next_token_log_probs[seq_has_eos & ~(next_tokens==eos_token)] = -np.log(10000)
        sequence_log_probs = torch.repeat_interleave(sequence_log_probs, len(tokenizer), 0)
        next_sequence_log_probs = sequence_log_probs + next_token_log_probs
        
        top_indicies = torch.argsort(next_sequence_log_probs, dim=0, descending=True).squeeze()[:n_beams]
        
        sequences = next_sequences[top_indicies]
        sequence_log_probs = next_sequence_log_probs[top_indicies]
    
    sequences = sequences.data.cpu().numpy()
    sequence_log_probs = sequence_log_probs.data.cpu().numpy()
    return sequences, sequence_log_probs



def compute_full_target_str(base_10_number, input_padding, max_encode_size, max_decode_size):
    number = dec2bin(base_10_number)
    factors = {dec2bin(k) : v for k, v in gfm[base_10_number].items()}
    tmp_ds = FactorizationDataset({0:{'number' : number, 'factors' : factors}}, max_encode_size, max_decode_size, input_padding)
    return tmp_ds[0]['label']
        
def postprocess(factor_list, log_prob, base_10_number, number, beam_idx, input_padding, max_encode_size, max_decode_size, tokenizer):
    tokenized = tokenizer(factor_list)
    
    information = {
        'target_num' : base_10_number,
        'target_is_prime' : gfm.is_prime(base_10_number),
        'target_str_full' : compute_full_target_str(base_10_number, input_padding, max_encode_size, max_decode_size),
        'target_factor_list' : sum([[k]*v for k, v in gfm[base_10_number].items()], []),
        'pred_str_full' : ''.join(tokenizer(factor_list, decode_special=True)) + '_'*(max_decode_size-len(factor_list)),
        'pred_str' : ''.join(tokenized),
        'beam_idx' : beam_idx,
        'log_prob' : log_prob.item(),
    }
    information['n_target_factors'] = len(information['target_factor_list'])
    information['n_target_zeros'] = information['target_str_full'].count('0')
    information['n_target_ones'] = information['target_str_full'].count('1')
    information['pct_target_ones'] = information['n_target_ones'] / (information['n_target_ones'] + information['n_target_zeros'])
    
    factor_list = tokenized.split('x')
    try:
        factors = [bin2dec(num) for num in factor_list]
    except ValueError:
        factors = []
    
    information['pred_factor_list'] = factors
    information['n_pred_factors'] = len(information['pred_factor_list'])
    if len(information['pred_factor_list']) > 0:
        information['product'] = np.prod(factors)
    else:
        information['product'] = np.nan
    information['correct_product'] = information['product']==base_10_number
    information['correct_factorization'] = information['correct_product'] & all([gfm.is_prime(n) for n in information['pred_factor_list']])
    
    information['num_prime_factors_pred'] = np.sum([gfm.is_prime(f) for f in factors if f in gfm])
    information['percent_prime_factors_pred'] = information['num_prime_factors_pred'] / information['n_pred_factors']
        
    
    target_str_full = information['target_str_full']
    pred_str_full = information['pred_str_full']
    information['seq_dist_binary'] = 1-np.mean([c1==c2 for c1, c2 in zip(target_str_full, pred_str_full)])
    information['seq_dist_lev'] = levenshtein_distance(target_str_full, pred_str_full)
    return information
    

        
def factor(number, model, tokenizer, device, input_padding, max_encode_size, max_decode_size, n_beams=1, return_type='df'):
    base_10_num = number
    model.eval()
    
    # Conver the number to a tensor
    number = dec2bin(number)
    number = tokenizer(pad_input(number, max_encode_size, input_padding))
    number = torch.tensor(number).unsqueeze(0).to(device)
    # Encode the number
    with torch.no_grad():
        memory, memory_key_padding_mask = model.encode(number)
    # Decode!
    factors_list, log_probs = decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams)
    number = number.data.cpu().numpy()[0]
    to_return = []
    for i in range(len(factors_list)):
        to_return.append(postprocess(factors_list[i], log_probs[i], base_10_num, number, i, input_padding, max_encode_size, max_decode_size, tokenizer))
        
    if return_type=='df':
        return pd.DataFrame.from_dict(to_return)
    elif return_type=='dict':
        return to_return
    else:
        raise ValueError('got unexpected return type %s'%return_type)