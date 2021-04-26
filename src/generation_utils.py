import numpy as np
import torch
import pandas as pd
import data_utils
from data_utils import dec2base, base2dec, pad_input, FactorizationDataset
from Levenshtein import distance as levenshtein_distance

# This function could be optimized a bit. For convineint I create all possible sequences @ each step w/ repeat_interlave, which is n_beams*len(tokenizer) sequences
# becaue it makes indexing convinient.
# According to the pytorch memory profiler, this takes about 1/6 of the running time of the total function. 
''' THIS ONE IS WORKING!!!
def decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams):
    sequences = torch.tensor(tokenizer('1')).to(device).unsqueeze(0)
    sequence_log_probs = torch.tensor([[0]]).to(device)
    eos_token = tokenizer('.')[0]
    pad_token = tokenizer('_')[0]
    
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
        seq_has_eos = torch.argmax((sequences==eos_token).int(), dim=1).view(-1,1) > 0
        if torch.all(seq_has_eos):
            # undo repeat_interleave
            sequences = sequences[::len(tokenizer)]
            break

        # Compute all possible next log probabilities
        next_token_log_probs[seq_has_eos & ~(next_tokens==pad_token)] = -np.log(10000)
        sequence_log_probs = torch.repeat_interleave(sequence_log_probs, len(tokenizer), 0)
        next_sequence_log_probs = sequence_log_probs + next_token_log_probs
        
        top_indicies = torch.argsort(next_sequence_log_probs, dim=0, descending=True).squeeze()[:n_beams]
        
        sequences = next_sequences[top_indicies]
        sequence_log_probs = next_sequence_log_probs[top_indicies]
    
    sequences = sequences.data.cpu().numpy()
    sequence_log_probs = sequence_log_probs.data.cpu().numpy()
    return sequences, sequence_log_probs
'''


def decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams):
    sequences = torch.tensor(tokenizer('>')).to(device).unsqueeze(0)
    sequence_log_probs = torch.tensor([[0]]).to(device)
    eos_token = tokenizer('.')[0]
    pad_token = tokenizer('_')[0]

    # Never decode start of sqeuence token
    n_valid_decode_tokens = len(tokenizer)-1
    
    for i in range(max_decode_size-1):
        with torch.no_grad():
            output = model.decode(sequences, 
                                  memory.repeat(1,sequences.size(0),1), 
                                  memory_key_padding_mask.repeat(sequences.size(0), 1))
            # Last timestep, ignore the start of sequence token b/c we never wanna predict it
            output = output[:,-1,:-1]
            output = torch.log_softmax(output, dim=-1)
        
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

def compute_full_target_str(base_10_number, base, input_padding, max_encode_size, max_decode_size):
    number = dec2base(base_10_number, base)
    factors = {dec2base(k, base) : v for k, v in data_utils.gfm[base_10_number].items()}
    tmp_ds = FactorizationDataset({0:{'number' : number, 'factors' : factors}}, max_encode_size, max_decode_size, input_padding)
    return tmp_ds[0]['label'].replace('_', '')
        
def postprocess(factor_list, log_prob, base_10_number, number, base, beam_idx, input_padding, max_encode_size, max_decode_size, tokenizer):
    tokenized = tokenizer(factor_list)
    
    information = {
        'target_num' : base_10_number,
        'target_is_prime' : data_utils.gfm.is_prime(base_10_number),
        'input_string' : dec2base(base_10_number, base),
        'target_str' : compute_full_target_str(base_10_number, base, input_padding, max_encode_size, max_decode_size),
        'target_factor_list' : sum([[k]*v for k, v in data_utils.gfm[base_10_number].items()], []),
        'pred_str' : ''.join(tokenized),
        'beam_idx' : beam_idx,
        'log_prob' : log_prob.item(),
    }
    information['n_target_factors'] = len(information['target_factor_list'])
    information['pred_same_as_target'] = information['input_string']==information['pred_str'][1:]
    
    factor_list = tokenized[1:].split('x')
    try:
        factors = [base2dec(num, base) for num in factor_list]
    except ValueError:
        factors = []
    
    information['pred_factor_list'] = factors
    information['n_pred_factors'] = len(information['pred_factor_list'])
    if len(information['pred_factor_list']) > 0:
        information['product'] = np.prod(factors)
    else:
        information['product'] = np.nan
    information['correct_product'] = information['product']==base_10_number
    information['correct_factorization'] = information['correct_product'] & all([data_utils.gfm.is_prime(n) for n in information['pred_factor_list']])
    
    information['num_prime_factors_pred'] = np.sum([data_utils.gfm.is_prime(f) for f in factors if f in data_utils.gfm])
    information['percent_prime_factors_pred'] = information['num_prime_factors_pred'] / information['n_pred_factors']
    
    return information
    

        
def factor(number, base, model, tokenizer, device, input_padding, max_encode_size, max_decode_size, n_beams=1, return_type='df'):
    base_10_num = number
    model.eval()
    
    # Conver the number to a tensor
    number = dec2base(number, base)
    # number = tokenizer(pad_input(number, max_encode_size, input_padding))
    number = tokenizer(number + '.')
    number = torch.tensor(number).unsqueeze(0).to(device)
    # Encode the number
    with torch.no_grad():
        memory, memory_key_padding_mask = model.encode(number)
    # Decode!
    factors_list, log_probs = decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams)
    number = number.data.cpu().numpy()[0]
    to_return = []
    for i in range(len(factors_list)):
        to_return.append(postprocess(factors_list[i], log_probs[i], base_10_num, number, base, i, input_padding, max_encode_size, max_decode_size, tokenizer))
        
    if return_type=='df':
        return pd.DataFrame.from_dict(to_return)
    elif return_type=='dict':
        return to_return
    else:
        raise ValueError('got unexpected return type %s'%return_type)