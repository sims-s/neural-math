import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



def visualize_attention(input_toks, tgt_toks, model, tokenizer, device):
    input_toks = torch.tensor(input_toks).unsqueeze(0).to(device)
    tgt_toks = torch.tensor(tgt_toks).unsqueeze(0).to(device)


    (memory, encoder_attn_weights), memory_key_padding_mask = model.encode(input_toks, need_weights=True)
    res, mem_attn, self_attn = model.decode(tgt_toks, memory.repeat(1, tgt_toks.size(0), 1), memory_key_padding_mask.repeat(tgt_toks.size(0), 1), return_enc_dec_attn=True)


    mem_label = tokenizer.decode(input_toks[0].data.cpu().numpy().tolist(), decode_special=True).split(' ')
    tgt_label_attended_to = tokenizer.decode(tgt_toks[0].data.cpu().numpy().tolist(), decode_special=True).split(' ')
    tgt_label_attended_for = tokenizer.decode(tgt_toks[0].data.cpu().numpy().tolist(), decode_special=True).split(' ')


    def show_attn(fig, ax, matrix, attn_type, title):
        ax.set_title(title)
        
        ax.set_ylabel('Predicting the next token')
        ax.set_xlabel('Attending to this token')
        
        if attn_type=='encoder_self':
            ax.set_yticks(np.arange(len(mem_label)))
            ax.set_yticklabels(labels=mem_label, fontsize=16)
            
            ax.set_xticks(np.arange(len(mem_label)))
            ax.set_xticklabels(labels=mem_label, fontsize=16, rotation=90)
        elif attn_type=='decoder_self':
            ax.set_yticks(np.arange(len(tgt_label_attended_for)))
            ax.set_yticklabels(labels=tgt_label_attended_for, fontsize=16)
            
            ax.set_xticks(np.arange(len(tgt_label_attended_to)))
            ax.set_xticklabels(labels=tgt_label_attended_to, fontsize=16, rotation=90)
        elif attn_type=='mem':
            ax.set_yticks(np.arange(len(tgt_label_attended_for)))
            ax.set_yticklabels(labels=tgt_label_attended_for, fontsize=16)
            
            ax.set_xticks(np.arange(len(mem_label)))
            ax.set_xticklabels(labels=mem_label, fontsize=16, rotation=90)
        else:
            raise ValueError(f'attn type {attn_type} not understood')

        
        
        im = ax.imshow(matrix, cmap='Blues')
        fig.colorbar(im, ax=ax)

    fig, ax = plt.subplots(encoder_attn_weights.size(0), encoder_attn_weights.size(1))
    for i in range(encoder_attn_weights.size(0)):
        for j in range(encoder_attn_weights.size(1)):
            title = '%s Layer: %d Head: %d'%('Encoder SA', i,j)
            show_attn(fig, ax[i,j], encoder_attn_weights[i][j].data.cpu().numpy(), 'encoder_self', title)
    fig.set_size_inches(36,36)
    fig.tight_layout()


    fig, ax = plt.subplots(mem_attn.size(0), mem_attn.size(1))
    for i in range(mem_attn.size(0)):
        for j in range(mem_attn.size(1)):
            title = '%s Layer: %d Head: %d'%('Mem', i,j)
            show_attn(fig, ax[i,j], mem_attn[i][j].data.cpu().numpy(), 'mem', title)
    fig.set_size_inches(36,36)
    fig.tight_layout()

    fig, ax = plt.subplots(self_attn.size(0), self_attn.size(1))
    for i in range(self_attn.size(0)):
        for j in range(self_attn.size(1)):
            title = '%s AttentionLayer: %d Head: %d'%('Self', i,j)
            show_attn(fig, ax[i,j], np.clip(self_attn[i][j].data.cpu().numpy(), a_min=0, a_max=1), 'decoder_self', title)
    fig.set_size_inches(36,36)
    fig.tight_layout()



