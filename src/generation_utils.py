import numpy as np
import torch




def decode(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams, temperature):
    # note: the start of sequence is always the final token; assumed in several places
    sequences = torch.tensor(tokenizer.encode(['[SOS]'])).to(device).unsqueeze(0)
    sequence_log_probs = torch.tensor([[0]]).to(device)
    eos_token = tokenizer.encode(['[EOS]'])[0]
    pad_token = tokenizer.encode(['[PAD]'])[0]

    # Never decode start of sqeuence token
    n_valid_decode_tokens = len(tokenizer)-1
    possible_decode_tokens = torch.arange(n_valid_decode_tokens, device = device)
    # non_pad_valid_tokens = torch.tensor(np.r_[:pad_token,pad_token+1:n_valid_decode_tokens], device=device)
    tok_non_pad = (possible_decode_tokens != pad_token)
    
    for i in range(max_decode_size-1):
        # print('BEAM IDX: ', i+1)
        # If every sequence has an end of sequence token, we're donet
        seq_has_eos = torch.argmax((sequences==eos_token).int(), dim=1) > 0
        if torch.all(seq_has_eos):
            break
        with torch.no_grad():
            output = model.decode(sequences, 
                                  memory.repeat(1,sequences.size(0),1), 
                                  memory_key_padding_mask.repeat(sequences.size(0), 1))
            # Last timestep, ignore the start of sequence token b/c we never wanna predict it
            output = output[:,-1,:-1]
            # if seq_has_eos.sum():
            #     mask_ = tok_non_pad.unsqueeze(0).repeat(seq_has_eos.size(0), 1)
            #     mask_[~seq_has_eos] = False
            #     output[mask_] = float('-inf')
            #     output[seq_has_eos,pad_token] = 1

            # if seq_has_eos.sum():    
            # # print('='*20)
            # # print('next token log probs size: ', next_token_log_probs.size())
            # # print('seq_has_eos/non_pad_valid size: ', seq_has_eos.size(), non_pad_valid_tokens.size())
            # # print(f'SEQ HAS EOS: ({seq_has_eos.sum().item()})', seq_has_eos)
            # # print('tok is pad size: ', tok_non_pad.size())
            # mask_ = tok_non_pad.unsqueeze(0).repeat(seq_has_eos.size(0), 1)
            # mask_[~seq_has_eos] = False
            # # print(next_token_log_probs.size(), mask_.size())
            # next_token_log_probs[mask_] = -np.log(10000)
            # next_token_log_probs[seq_has_eos, pad_token] = 0


            next_token_log_probs = torch.log_softmax(output / temperature, dim=-1)
        
        # Compue all possible next sequences
        # sequences = torch.repeat_interleave(sequences, len(tokenizer), 0)
        # next_tokens = torch.arange(n_valid_decode_tokens, device=sequences.device).repeat(output.size(0)).unsqueeze(1)
        # next_sequences = torch.cat((sequences, next_tokens), dim=1)
        
        # Get the indiclies of the highest probability sequences
        # next_token_log_probs = output.view(-1,1)
        # For cases when the model migth predict a non padding token after an end of sequence token
        # Manually set the log probability to be very low so it's never chosen to be decdoed
        # Additionally, break if all sequences have eos tokens
        

        # Set the probabilities of decoding a non-padding token after EOS tokens to be small
        
            # print(sequences[~seq_has_eos])
        """
        print(seq_has_eos.size(), possible_decode_tokens.size(), next_token_log_probs.size())
        print(( ~(possible_decode_tokens == pad_token)).size())
        print(non_pad_valid_tokens)
        print(next_token_log_probs[seq_has_eos].size())
        print(next_token_log_probs[:,non_pad_valid_tokens].size())
        next_token_log_probs[seq_has_eos, non_pad_valid_tokens] = -np.log(10000)
        sys.exit()
        if seq_has_eos.sum():
            # next_token_log_probs[seq_has_eos, ~(possible_decode_tokens == pad_token)] = -np.log(10000)
            next_token_log_probs[seq_has_eos, non_pad_valid_tokens] = -np.log(10000)
        """

        if seq_has_eos.sum():    
            # print('='*20)
            # print('next token log probs size: ', next_token_log_probs.size())
            # print('seq_has_eos/non_pad_valid size: ', seq_has_eos.size(), non_pad_valid_tokens.size())
            # print(f'SEQ HAS EOS: ({seq_has_eos.sum().item()})', seq_has_eos)
            # print('tok is pad size: ', tok_non_pad.size())
            mask_ = tok_non_pad.unsqueeze(0).repeat(seq_has_eos.size(0), 1)
            mask_[~seq_has_eos] = False
            # print(next_token_log_probs.size(), mask_.size())
            next_token_log_probs[mask_] = -np.log(10000)
            next_token_log_probs[seq_has_eos, pad_token] = 0
            # print(next_token_log_probs[seq_has_eos][:,non_pad_valid_tokens].size())
            # print(next_token_log_probs[seq_has_eos][:,non_pad_valid_tokens])
            # print('='*20)
            # sys.exit()



        # sequence_log_probs = torch.repeat_interleave(sequence_log_probs, n_valid_decode_tokens, 0)
        # sequence log probs: (1 for first iter OR n_beams for later iters x 1)
        # next token log probs: (1 for first iter OR nbeams for later iters x n_valid_tokens)
        # print('seq log probs size: ', sequence_log_probs.size())
        # print('next token log probs size: ', next_token_log_probs.size())
        next_sequence_log_probs = sequence_log_probs + next_token_log_probs
        # print('next sequence log probs size: ', next_sequence_log_probs.size())
        flat_lps = next_sequence_log_probs.view(-1)
        top_indices = flat_lps.argsort(descending=True)[:n_beams]
        beam_indices = top_indices // n_valid_decode_tokens
        tok_indices = top_indices % n_valid_decode_tokens
        # print(top_indices)
        # print(beam_indices, tok_indices)
        # print(flat_lps[top_indices])
        
        # top_indicies = torch.argsort(next_sequence_log_probs, dim=0, descending=True).squeeze()[:n_beams]
        # print(sequences[beam_indices].size(), possible_decode_tokens[tok_indices].size())
        sequences = torch.cat([sequences[beam_indices], possible_decode_tokens[tok_indices].unsqueeze(1)], dim=1)
        # print(beam_indices.size(), tok_indices.size(), next_sequence_log_probs.size())
        # print(next_sequence_log_probs[beam_indices][:,tok_indices].size())
        sequence_log_probs = next_sequence_log_probs[beam_indices,tok_indices].unsqueeze(1)

        # print('final sequences size: ', sequences.size())
        # print('final logprobs size: ', sequence_log_probs.size())
        # print(sequences)
        # print(sequence_log_probs)
    #     print('='*100)
    #     print('='*100)
    #     print('='*100)
    # print(sequences)
    # print(sequence_log_probs)
    # sys.exit()
    
    sequences = sequences.data.cpu().numpy()
    sequence_log_probs = sequence_log_probs.data.cpu().numpy()
    return sequences, sequence_log_probs



# decode, but old. Couldn't resist decold.... ;)
def decold(model, tokenizer, device, max_decode_size, memory, memory_key_padding_mask, n_beams, temperature):
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
