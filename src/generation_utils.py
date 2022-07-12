import numpy as np
import torch




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
        next_tokens = torch.arange(n_valid_decode_tokens, device=sequences.device).repeat(output.size(0)).unsqueeze(1)
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



# """================="""
# """FOR FACTORIZATION"""
# """================="""

    
            
    
        
# def factor(number, base, model, tokenizer, device, max_decode_size, n_beams=1, temperature=1.0, return_type='df', postprocess_minimal=False):
#     base_10_num = number
#     model.eval()
    
#     # Conver the number to a tensor
#     number = tokenizer.encode(data_utils.form_input(number, base))
#     number = torch.tensor(number).unsqueeze(0).to(device)
#     # Encode the number
#     with torch.no_grad():
#         memory, memory_key_padding_mask = model.encode(number)
#     # Decode!
#     factors_list, log_probs = decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams, temperature)
#     number = number.data.cpu().numpy()[0]
#     to_return = []
#     for i in range(len(factors_list)):
#         to_return.append(postprocess(factors_list[i], log_probs[i], base_10_num, base, i, tokenizer, postprocess_minimal))
        
#     if return_type=='df':
#         return pd.DataFrame.from_dict(to_return)
#     elif return_type=='dict':
#         return to_return
#     else:
#         raise ValueError('got unexpected return type %s'%return_type)



# """============"""
# """FOR ADDITION"""
# """============"""



# def model_add(n1, n2, base, model, tokenizer, device, max_decode_size, n_beams, temperature, return_type='dict'):
#     base_10_n1 = n1
#     base_10_n2 = n2
#     input_list = data_utils.form_input_addition(n1, n2, base)
#     label_list = data_utils.form_label_addition(n1, n2, base)

#     input = torch.tensor(tokenizer.encode(input_list), device=device).unsqueeze(0)

#     with torch.no_grad():
#         memory, memory_key_padding_mask = model.encode(input)

#     outputs, logprobs = decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams, temperature)

#     to_return = []
#     for i in range(len(outputs)):
#         number_predicted = decode_beam_addition(outputs[i], tokenizer, base)
#         add_dict = {
#             'n1' : base_10_n1, 
#             'n2' : base_10_n2, 
#             'n1 + n2' : base_10_n1 + base_10_n2, 
#             'input_list': input_list,
#             'label_list' : label_list,
#             'input_str' : ''.join([str(c) for c in input_list]),
#             'label_str' : ''.join([str(c) for c in label_list]),
#             'pred_tokens' : outputs[i].tolist(),
#             'pred_num' : number_predicted,
#             'log_prob' : logprobs[i].item(),
#             'beam_idx' : i
#         }
#         if add_dict['pred_num'] is None:
#             add_dict['pred_is_right'] = False
#         else:
#             add_dict['pred_is_right'] = int(n1 + n2) == int(add_dict['pred_num'])
#         to_return.append(add_dict)
        

#     if return_type=='df':
#         return pd.DataFrame.from_dict(to_return)
#     elif return_type=='dict':
#         return to_return
#     else:
#         raise ValueError('got unexpected return type %s'%return_type)
