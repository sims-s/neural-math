from os import X_OK
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules.transformer import Transformer
from attention_utils import MultiHeadAttention, MultiHeadRelativeAttention
import sys
sys.path.append('src/transformer_generalization/layers/transformer/')
from multi_head_relative_pos_attention import FixedRelativeMultiheadAttention

class PositionalEncoding(nn.Module):
    # From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout, max_len, learnable, pos_offset=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) + pos_offset
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        if learnable:
            self.pe = nn.Parameter(pe)
        else:
            self.register_buffer('pe', pe)

    def forward(self, x, pos_offset=0):
        # print('Positional encodings get called. Input size: ', x.size())
        # print('overall PE size: ', self.pe.size())
        # print('PE addition size: ', self.pe[:x.size(0), :].size())
        x = x + self.pe[pos_offset : x.size(0) + pos_offset, :]
        # x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, n_tokens, embed_dim, scale_embeddings, scale_embeddings_at_init):
        super(TransformerEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_tokens, embed_dim)
        if scale_embeddings_at_init:
            self.embedding.weight.data = self.embedding.weight * math.sqrt(self.embed_dim)
        self.scale_embeddings_realtime = scale_embeddings and not scale_embeddings_at_init

    def forward(self, x):
        embedded = self.embedding(x)
        if self.scale_embeddings_realtime:
            embedded = embedded * math.sqrt(self.embed_dim)
        return embedded.transpose(0,1)


class Factorizer(nn.Module):
    def __init__(self,  n_tokens, 
                        embed_dim, 
                        pad_token_id, 
                        max_decode_size=64, 
                        shared_embeddings=True,
                        scale_embeddings=False,
                        scale_embeddings_at_init=False, 
                        learn_positional_encoding=False, 
                        repeat_positional_encoding=False, 
                        positional_encoding_query_key_only=False, 
                        norm_first = False, 
                        relative_positional_encoding = False,
                         **kwargs):
        super(Factorizer, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.pad_token_id = pad_token_id
        self.repeat_positional_encoding = repeat_positional_encoding
        self.positional_encoding_query_key_only = positional_encoding_query_key_only
        self.norm_first = norm_first
        self.relative_positional_encoding = relative_positional_encoding

        num_heads = 8 if not 'num_heads' in kwargs else kwargs['num_heads']
        dropout = .1 if not 'dropout' in kwargs else kwargs['dropout']

        if not shared_embeddings:
            self.src_embedding = TransformerEmbedding(n_tokens, embed_dim, scale_embeddings, scale_embeddings_at_init)
            self.tgt_embedding = TransformerEmbedding(n_tokens, embed_dim, scale_embeddings, scale_embeddings_at_init)
        else:
            self.embedding = TransformerEmbedding(n_tokens, embed_dim, scale_embeddings, scale_embeddings_at_init)

        if self.relative_positional_encoding:
            self.positional_encoding = PositionalEncoding(embed_dim, dropout, 2*max_decode_size-1, learn_positional_encoding, -max_decode_size+1)
        else:
            self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_decode_size, learn_positional_encoding)
        
        self.transformer = nn.Transformer(d_model = embed_dim, **kwargs)
        
        if self.relative_positional_encoding:
            for i in range(self.transformer.encoder.num_layers):
                self.transformer.encoder.layers[i].self_attn = MultiHeadRelativeAttention(embed_dim, num_heads, self.positional_encoding, dropout)
        for i in range(self.transformer.decoder.num_layers):
            if self.relative_positional_encoding:
                self.transformer.decoder.layers[i].self_attn = MultiHeadRelativeAttention(embed_dim, num_heads, self.positional_encoding, dropout)
            else:
                self.transformer.decoder.layers[i].self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
            self.transformer.decoder.layers[i].multihead_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.tokens_out = nn.Linear(embed_dim, n_tokens)
        
    
    def form_pad_mask(self, tokens):
        return (tokens==self.pad_token_id).to(tokens.device)
    
    def form_subsequence_mask(self, tgt):
        size = tgt.size(0)
        return torch.triu(torch.ones(size, size), 1).bool().to(tgt.device)
    
    def route_embeddings(self, src_or_tgt, input_type):
        if self.shared_embeddings:
            return self.embedding(src_or_tgt)
        if input_type=='src':
            return self.src_embedding(src_or_tgt)
        elif input_type=='tgt':
            return self.tgt_embedding(src_or_tgt)
    
    def encode(self, src):
        src_key_padding_mask = self.form_pad_mask(src)
        src = self.route_embeddings(src, 'src')
        
        return self.encoder_forward(src, src_key_padding_mask = src_key_padding_mask), src_key_padding_mask

    def prepare_embedding_for_attn(self, emb, is_first_mod, mod_is_relative):
        if mod_is_relative:
            q = emb
            k = emb
            v = emb
        elif not self.repeat_positional_encoding and is_first_mod:
            emb_with_pe = self.positional_encoding(emb)
            q = emb_with_pe
            k = emb_with_pe
            v = emb_with_pe
        elif not self.repeat_positional_encoding and not is_first_mod:
            q = emb
            k = emb
            v = emb
        elif self.repeat_positional_encoding:
            emb_with_pe = self.positional_encoding(emb)
            q = emb_with_pe
            k = emb_with_pe
            if self.positional_encoding_query_key_only:
                v = emb
            else:
                v = emb_with_pe
        
        return q, k, v

    
    def encoder_mod_forward(self, x, mod, src_mask, src_key_padding_mask, is_first_enc_mod):
        mod_is_relative = isinstance(mod.self_attn, MultiHeadRelativeAttention)
        # print('Call forward on an encoder module')
        # print('module is first encoder: ', is_first_enc_mod)
        def _sa_block(x, src_mask, src_key_padding_mask):
            q, k, v = self.prepare_embedding_for_attn(x, is_first_enc_mod, mod_is_relative)
            x = mod.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
            return mod.dropout1(x)

        def _ff_block(x):
            x = mod.linear2(mod.dropout(mod.activation(mod.linear1(x))))
            return mod.dropout2(x)
        
        if self.norm_first:
            x = x + _sa_block(mod.norm1(x), src_mask, src_key_padding_mask)
            x = x + _ff_block(mod.norm2(x))
        else:
            x = mod.norm1(x + _sa_block(x, src_mask, src_key_padding_mask))
            x = mod.norm2(x + _ff_block(x))
        return x


    def encoder_forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        for i, mod in enumerate(self.transformer.encoder.layers):
            x = self.encoder_mod_forward(x, mod, src_mask, src_key_padding_mask, i==0)

        if self.transformer.encoder.norm is not None:
            x = self.transformer.encoder.norm(x)
        return x

    def decoder_layer_forward_with_attention(self, mod, x, memory, tgt_mask=None, tgt_key_padding_mask=None, 
                                            mem_mask=None, memory_key_padding_mask=None, is_first_dec_mod=False):
        mod_is_relative_self = isinstance(mod.self_attn, MultiHeadRelativeAttention)
        # mod_is_relative_mha = isinstance(mod.multihead_attn, MultiHeadRelativeAttention)
        def _sa_block(x, attn_mask, key_padding_mask):
            q, k, v = self.prepare_embedding_for_attn(x, is_first_dec_mod, mod_is_relative_self)
            x, attn_weights = mod.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
            return mod.dropout1(x), attn_weights

        def _mha_block(x, mem, attn_mask, key_padding_mask):
            if self.relative_positional_encoding:
                q = x
                k = mem
                v = mem
                # offset = self.positional_encoding.pe.size(0) // 2 + 1
                # q = self.positional_encoding(x, pos_offset=offset)
                # k = self.positional_encoding(mem, pos_offset=offset)
                # if not self.positional_encoding_query_key_only:
                #     v = self.positional_encoding(mem, pos_offset=offset)
                # else:
                #     v = mem
            elif self.repeat_positional_encoding:
                # IN RETROSPECT, I THINK THIS IS WRONG? BUT I NEED TO DO A COMPARISON
                # don't repositionally encode the query, since it already has p.e.s from self attn layer?
                # q = self.positional_encoding(x)
                q = x
                k = self.positional_encoding(mem)
                if not self.positional_encoding_query_key_only:
                    v = self.positional_encoding(mem)
                else:
                    v = mem
            else:
                q = x
                k = mem
                v = mem

            x, attn_weights = mod.multihead_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
            return mod.dropout2(x), attn_weights

        def _ff_block(x):
            x = mod.linear2(mod.dropout(mod.activation(mod.linear1(x))))
            return mod.dropout2(x)

        if self.norm_first:
            sa_result, self_attn_weight = _sa_block(mod.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + sa_result
            mha_result, mha_attn_weight = _mha_block(mod.norm2(x), memory, mem_mask, memory_key_padding_mask)
            x = x + mha_result
            x = x + _ff_block(mod.norm3(x))
        else:
            sa_result, self_attn_weight = _sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = mod.norm1(x + sa_result)
            mha_result, mha_attn_weight = _mha_block(x, memory, mem_mask, memory_key_padding_mask)
            x = mod.norm2(x + mha_result)
            x = mod.norm3(x + _ff_block(x))
        
        return x, self_attn_weight, mha_attn_weight



    def decoder_forward_with_attention(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        x = tgt
        mem_attn_list = []
        self_attn_list = []
        for i, mod in enumerate(self.transformer.decoder.layers):
            x, self_attn, mem_attn = self.decoder_layer_forward_with_attention(mod, x, memory, tgt_mask=tgt_mask, 
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask, is_first_dec_mod=i==0)
            mem_attn_list.append(mem_attn)
            self_attn_list.append(self_attn)

        if self.transformer.decoder.norm is not None:
            x = self.transformer.decoder.norm(x)
        
        return x, torch.cat(mem_attn_list), torch.cat(self_attn_list)

    def decode(self, tgt, memory, memory_key_padding_mask, return_enc_dec_attn=False):
        tgt_key_padding_mask = self.form_pad_mask(tgt)
        tgt = self.route_embeddings(tgt, 'tgt')
        tgt_mask = self.form_subsequence_mask(tgt)
        
        output, mem_attn, self_attn = self.decoder_forward_with_attention(tgt, memory, tgt_mask=tgt_mask,
                tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = memory_key_padding_mask)

        output = output.transpose(0,1)
        decoded = self.tokens_out(output)
        if not return_enc_dec_attn:
            return decoded
        else:
            return decoded, mem_attn, self_attn
    
        
    def forward(self, src, tgt):
        memory, memory_key_padding_mask = self.encode(src)
        decoded = self.decode(tgt, memory, memory_key_padding_mask)

        return decoded