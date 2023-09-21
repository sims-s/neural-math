import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



def visualize_attention(input_toks, tgt_toks, model=None, tokenizer=None, device=None,
                        ee_attn = None, ed_attn = None, dd_attn = None):
    
    if ee_attn is None:
        input_toks = torch.tensor(input_toks, device=device).unsqueeze(0)
        tgt_toks = torch.tensor(tgt_toks, device=device).unsqueeze(0)


        (memory, ee_attn), memory_key_padding_mask = model.encode(input_toks, need_weights=True)
        res, ed_attn, dd_attn = model.decode(tgt_toks, memory.repeat(1, tgt_toks.size(0), 1), memory_key_padding_mask.repeat(tgt_toks.size(0), 1), return_enc_dec_attn=True)
    else:
        assert ed_attn is not None
        assert dd_attn is not None


    mem_label = tokenizer.decode(input_toks[0].data.cpu().numpy().tolist(), decode_special=True).split(' ')
    tgt_label_attended_to = tokenizer.decode(tgt_toks[0].data.cpu().numpy().tolist(), decode_special=True).split(' ')
    tgt_label_attended_for = tokenizer.decode(tgt_toks[0].data.cpu().numpy().tolist(), decode_special=True).split(' ')


    def show_attn(fig, ax, matrix, attn_type, title):
        ax.set_title(title)
        
        ax.set_ylabel('Attending For this token')
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
        fig.tight_layout()

    fig, ax = plt.subplots(ee_attn.size(0), ee_attn.size(1))
    for i in range(ee_attn.size(0)):
        for j in range(ee_attn.size(1)):
            title = '%s Layer: %d Head: %d'%('EncEnc', i,j)
            show_attn(fig, ax[i,j], ee_attn[i][j].data.cpu().numpy(), 'encoder_self', title)
    fig.set_size_inches(36,36)
    fig.tight_layout()


    fig, ax = plt.subplots(ed_attn.size(0), ed_attn.size(1))
    for i in range(ed_attn.size(0)):
        for j in range(ed_attn.size(1)):
            title = '%s Layer: %d Head: %d'%('EncDec', i,j)
            show_attn(fig, ax[i,j], ed_attn[i][j].data.cpu().numpy(), 'mem', title)
    fig.set_size_inches(36,36)
    fig.tight_layout()

    fig, ax = plt.subplots(dd_attn.size(0), dd_attn.size(1))
    for i in range(dd_attn.size(0)):
        for j in range(dd_attn.size(1)):
            title = '%s Layer: %d Head: %d'%('DecDec', i,j)
            show_attn(fig, ax[i,j], np.clip(dd_attn[i][j].data.cpu().numpy(), a_min=0, a_max=1), 'decoder_self', title)
    fig.set_size_inches(36,36)
    fig.tight_layout()



