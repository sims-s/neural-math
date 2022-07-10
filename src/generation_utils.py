import numpy as np
import torch
import pandas as pd
import data_utils
import utils
from data_utils import dec2base, base2dec, FactorizationDataset
from Levenshtein import distance as levenshtein_distance
from sympy import factorint
from sympy.ntheory.primetest import isprime


"""================="""
"""FOR FACTORIZATION"""
"""================="""
def decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams, temperature):
    sequences = torch.tensor(tokenizer.encode('>')).to(device).unsqueeze(0)
    sequence_log_probs = torch.tensor([[0]]).to(device)
    eos_token = tokenizer.encode('.')[0]
    pad_token = tokenizer.encode('_')[0]

    # Never decode start of sqeuence token
    n_valid_decode_tokens = len(tokenizer)-1
    
    for i in range(max_decode_size-1):
        with torch.no_grad():
            output = model.decode(sequences, 
                                  memory.repeat(1,sequences.size(0),1), 
                                  memory_key_padding_mask.repeat(sequences.size(0), 1))
            # Last timestep, ignore the start of sequence token b/c we never wanna predict it
            output = output[:,-1,:-1]
            output = torch.log_softmax(output / temperature, dim=-1)
        # Compue all possible next sequences
        # sequences = torch.repeat_interleave(sequences, len(tokenizer), 0)
        next_tokens = torch.arange(n_valid_decode_tokens).repeat(output.size(0)).unsqueeze(1).to(sequences.device)
        # next_sequences = torch.cat((sequences, next_tokens), dim=1)
        
        # Get the indiclies of the highest probability sequences
        next_token_log_probs = output.view(-1,1)
        # For cases when the model migth predict a non padding token after an end of sequence token
        # Manually set the log probability to be very low so it's never chosen to be decdoed
        # Additionally, break if all sequences have eos tokens
        seq_has_eos = torch.argmax((sequences==eos_token).int(), dim=1).view(-1,1) > 0
        if torch.all(seq_has_eos):
            break

        # Compute all possible next log probabilities
        next_token_log_probs[torch.repeat_interleave(seq_has_eos, n_valid_decode_tokens, 0) & ~(next_tokens==pad_token)] = -np.log(10000)
        sequence_log_probs = torch.repeat_interleave(sequence_log_probs, n_valid_decode_tokens, 0)
        next_sequence_log_probs = sequence_log_probs + next_token_log_probs
        
        top_indicies = torch.argsort(next_sequence_log_probs, dim=0, descending=True).squeeze()[:n_beams]

        sequences = torch.cat((sequences[top_indicies//n_valid_decode_tokens], next_tokens[top_indicies]), dim=1)
        sequence_log_probs = next_sequence_log_probs[top_indicies]
    
    sequences = sequences.data.cpu().numpy()
    sequence_log_probs = sequence_log_probs.data.cpu().numpy()
    return sequences, sequence_log_probs


def extract_factors(factor_list):
    chunked = []
    chunk_start_idx = 0
    for i, token in enumerate(factor_list):
        if token=='x' or token =='.':
            chunked.append(factor_list[chunk_start_idx:i])
            chunk_start_idx = i + 1
        elif i==len(factor_list)-1:
            chunked.append(factor_list[chunk_start_idx:i+1])
    return chunked
    
            
        
def postprocess(factor_list, log_prob, base_10_number, base, beam_idx, tokenizer, postprocess_minimal):
    information = {
        'input_num' : base_10_number,
            'model_input' : [str(c) for c in data_utils.form_input(base_10_number, base)],
        'beam_idx' : beam_idx,
        'log_prob' : log_prob.item(),
    }

    tokenized = tokenizer.decode(factor_list, decode_special=True)
    factor_list = utils.drop_from_iterable(tokenized.split(' '), ['>', '_', '.'])
    factor_list = extract_factors(factor_list)

    try:
        factors = [base2dec([int(digit) for digit in num], base) for num in factor_list]
    except ValueError:
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
        information['target_factor_str'] = ' '.join([str(c) for c in data_utils.form_label(base_10_number, base)])
        information['target_factor_list'] = sum([[k]*v for k, v in factorint(base_10_number).items()], [])

        information['num_target_factors'] = len(information['target_factor_list'])

        information['min_target_prime_factor_if_composite'] = -1 if len(information['target_factor_list'])==1 else min(information['target_factor_list'])

    return information
        
def factor(number, base, model, tokenizer, device, max_decode_size, n_beams=1, temperature=1.0, return_type='df', postprocess_minimal=False):
    base_10_num = number
    model.eval()
    
    # Conver the number to a tensor
    number = tokenizer.encode(data_utils.form_input(number, base))
    number = torch.tensor(number).unsqueeze(0).to(device)
    # Encode the number
    with torch.no_grad():
        memory, memory_key_padding_mask = model.encode(number)
    # Decode!
    factors_list, log_probs = decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams, temperature)
    number = number.data.cpu().numpy()[0]
    to_return = []
    for i in range(len(factors_list)):
        to_return.append(postprocess(factors_list[i], log_probs[i], base_10_num, base, i, tokenizer, postprocess_minimal))
        
    if return_type=='df':
        return pd.DataFrame.from_dict(to_return)
    elif return_type=='dict':
        return to_return
    else:
        raise ValueError('got unexpected return type %s'%return_type)



"""============"""
"""FOR ADDITION"""
"""============"""
def decode_beam_addition(beam, tokenizer, base):
    sos = tokenizer.encode('>')[0]
    eos = tokenizer.encode('.')[0]
    pad = tokenizer.encode('_')[0]
    # Ignore all padding tokens
    beam = [tok for tok in beam if not tok==pad]
    start = None if not sos in beam else beam.index(sos)
    end = None if not eos in beam else beam.index(eos)
    if start is not None and end is not None:
        beam = beam[start+1:end]
        try:
            return data_utils.base2dec(beam, base)
        except (ValueError, TypeError):
            return None
    else:
        return None

from functools import lru_cache
@lru_cache
def model_add(n1, n2, base, model, tokenizer, device, max_decode_size, n_beams, temperature, return_type='dict'):
    base_10_n1 = n1
    base_10_n2 = n2
    input_list = data_utils.form_input_addition(n1, n2, base)
    label_list = data_utils.form_label_addition(n1, n2, base)

    input = torch.tensor(tokenizer.encode(input_list), device=device).unsqueeze(0)

    with torch.no_grad():
        memory, memory_key_padding_mask = model.encode(input)

    outputs, logprobs = decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams, temperature)

    to_return = []
    for i in range(len(outputs)):
        number_predicted = decode_beam_addition(outputs[i], tokenizer, base)
        add_dict = {
            'n1' : base_10_n1, 
            'n2' : base_10_n2, 
            'n1 + n2' : base_10_n1 + base_10_n2, 
            'input_list': input_list,
            'label_list' : label_list,
            'input_str' : ''.join([str(c) for c in input_list]),
            'label_str' : ''.join([str(c) for c in label_list]),
            'pred_tokens' : outputs[i].tolist(),
            'pred_num' : number_predicted,
            'log_prob' : logprobs[i].item(),
            'beam_idx' : i
        }
        if add_dict['pred_num'] is None:
            add_dict['pred_is_right'] = False
        else:
            add_dict['pred_is_right'] = int(n1 + n2) == int(add_dict['pred_num'])
        to_return.append(add_dict)
        

    if return_type=='df':
        return pd.DataFrame.from_dict(to_return)
    elif return_type=='dict':
        return to_return
    else:
        raise ValueError('got unexpected return type %s'%return_type)
