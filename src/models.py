from os import X_OK
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules.transformer import Transformer
from attention_utils import MultiHeadAttention, MultiHeadRelativeAttention

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
        x = x + self.pe[pos_offset : x.size(0) + pos_offset, :]
        return self.dropout(x)

# class PositionalEncodingDeberta(nn.Module):
#     def __init__(self, d_model, dropout, max_len, layernorm, position_buckets):
#         super(PositionalEncodingDeberta, self).__init__()
#         self.max_relative_positions = max_len

#         self.position_buckets = position_buckets
#         pos_ebd_size = self.max_relative_positions*2

#         if self.position_buckets>0:
#             pos_ebd_size = self.position_buckets*2
#         self.rel_embeddings = nn.Embedding(pos_ebd_size, d_model)
#         if layernorm:
#             self.layernorm = nn.LayerNorm(d_model, elementwise_affine=True)
#         else:
#             self.layernorm = None

#         self.dropout = nn.Dropout(p=dropout)

#     def get_rel_embeddings(self):
#         rel_embeddings = self.rel_embeddings.weight
#         if self.layernorm:
#             rel_embeddings = self.layernorm(rel_embeddings)
#         rel_embeddings = self.dropout(rel_embeddings)
#         return rel_embeddings

class TransformerEmbedding(nn.Module):
    def __init__(self, n_tokens, embed_dim, initialization, scale_embeddings, scale_embeddings_at_init):
        super(TransformerEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_tokens, embed_dim)
        if scale_embeddings_at_init:
            self.embedding.weight.data = self.embedding.weight * math.sqrt(self.embed_dim)
        self.scale_embeddings_realtime = scale_embeddings and not scale_embeddings_at_init
        self.initialization = initialization
        self._reset_parameters(initialization)

    def forward(self, x):
        embedded = self.embedding(x)
        if self.scale_embeddings_realtime:
            embedded = embedded * math.sqrt(self.embed_dim)
        return embedded.transpose(0,1)

    def _reset_parameters(self, init_type):
        init_type = init_type.lower()
        if init_type in ["normal"]:
            nn.init.normal_(self.embedding.weight)
        elif init_type in ['xavier', 'xavier_uniform', 'glorot']:
            nn.init.xavier_uniform_(self.embedding.weight)
        elif init_type in ['kaiming', 'kaiming_normal']:
            nn.init.kaiming_normal_(self.embedding.weight)
        


class Seq2SeqModel(nn.Module):
    def __init__(self,  n_tokens, 
                        embed_dim, 
                        pad_token_id, 
                        num_heads = 8,
                        dropout = .1,
                        max_decode_size=64, 
                        shared_embeddings=True,
                        scale_embeddings=False,
                        scale_embeddings_at_init=False, 
                        learn_positional_encoding=False, 
                        repeat_positional_encoding=False, 
                        positional_encoding_query_key_only=False, 
                        norm_first = False, 
                        positional_encoding_type = 'absolute',
                        extra_positional_encoding_relative_decoder_mha = False,
                        attn_weight_xavier_init_constant = .5,
                        embedding_initialization = 'xavier',
                        pad_embs_to_8_multiple = False,
                         **kwargs):
        super(Seq2SeqModel, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.pad_token_id = pad_token_id
        self.repeat_positional_encoding = repeat_positional_encoding
        self.positional_encoding_query_key_only = positional_encoding_query_key_only
        self.norm_first = norm_first
        valid_pe_types = ['absolute', 'relative-transfxl']
        if not positional_encoding_type in valid_pe_types:
            raise ValueError(f'unexpected positional encoding, {positional_encoding_type}, must be one of: ', valid_pe_types)
        
        self.positional_encoding_type = positional_encoding_type
        self.extra_positional_encoding_relative_decoder_mha = extra_positional_encoding_relative_decoder_mha

        self.num_heads = num_heads

        extra_toks = 0 if not pad_embs_to_8_multiple else n_tokens % 8
        self.src_embedding = TransformerEmbedding(n_tokens + extra_toks, embed_dim, embedding_initialization, scale_embeddings, scale_embeddings_at_init)
        self.tgt_embedding = TransformerEmbedding(n_tokens + extra_toks, embed_dim, embedding_initialization, scale_embeddings, scale_embeddings_at_init) \
                             if not shared_embeddings else self.src_embedding

        if self.positional_encoding_type=='relative-transfxl':
            self.positional_encoding = PositionalEncoding(embed_dim, dropout, 2*max_decode_size-1, learn_positional_encoding, -max_decode_size+1)
        elif self.positional_encoding_type=='absolute':
            self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_decode_size, learn_positional_encoding)

        self.transformer = nn.Transformer(d_model = embed_dim, nhead = self.num_heads, **kwargs)
        
        
        for i in range(self.transformer.encoder.num_layers):
            if self.positional_encoding_type=='relative-transfxl':
                self.transformer.encoder.layers[i].self_attn = MultiHeadRelativeAttention(embed_dim, num_heads, self.positional_encoding, dropout, attn_weight_xavier_init_constant)
            elif self.positional_encoding_type=='absolute':
                self.transformer.encoder.layers[i].self_attn = MultiHeadAttention(embed_dim, num_heads, dropout, attn_weight_xavier_init_constant)


        for i in range(self.transformer.decoder.num_layers):
            if self.positional_encoding_type=='relative-transfxl':
                self.transformer.decoder.layers[i].self_attn = MultiHeadRelativeAttention(embed_dim, num_heads, self.positional_encoding, dropout, attn_weight_xavier_init_constant)
            elif self.positional_encoding_type=='absolute':
                self.transformer.decoder.layers[i].self_attn = MultiHeadAttention(embed_dim, num_heads, dropout, attn_weight_xavier_init_constant)

            self.transformer.decoder.layers[i].multihead_attn = MultiHeadAttention(embed_dim, num_heads, dropout, attn_weight_xavier_init_constant)
        
        self.tokens_out = nn.Linear(embed_dim, n_tokens)
        
    
    def form_pad_mask(self, tokens):
        return (tokens==self.pad_token_id).to(tokens.device)
    
    def form_subsequence_mask(self, tgt):
        size = tgt.size(0)
        return torch.triu(torch.ones(size, size), 1).bool().to(tgt.device)
    
    def encode(self, src, need_weights=False):
        src_key_padding_mask = self.form_pad_mask(src)
        src = self.src_embedding(src)
        return self.encoder_forward(src, src_key_padding_mask = src_key_padding_mask, need_weights=need_weights), src_key_padding_mask

    def prepare_embedding_for_attn(self, emb, is_first_mod, use_pe):
        if not use_pe:
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

    
    def encoder_mod_forward(self, x, mod, src_mask, src_key_padding_mask, is_first_enc_mod, need_weights=False):
        use_pe = self.positional_encoding_type=='absolute'
        def _sa_block(x, src_mask, src_key_padding_mask):
            q, k, v = self.prepare_embedding_for_attn(x, is_first_enc_mod, use_pe)
            x, attn_weights = mod.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=True)
            return x, attn_weights
            

        def _ff_block(x):
            x = mod.linear2(mod.dropout(mod.activation(mod.linear1(x))))
            return mod.dropout2(x)
        
        if self.norm_first:
            sa_block_result, attn_weights = _sa_block(mod.norm1(x), src_mask, src_key_padding_mask)
            x = x + sa_block_result
            x = x + _ff_block(mod.norm2(x))
        else:
            sa_block_result, attn_weights = _sa_block(x, src_mask, src_key_padding_mask)
            x = mod.norm1(x + sa_block_result)
            x = mod.norm2(x + _ff_block(x))
        if need_weights:
            return x, attn_weights
        else:
            return x


    def encoder_forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=False):
        x = src
        if need_weights:
            all_weights = []
        for i, mod in enumerate(self.transformer.encoder.layers):
            if need_weights:
                x, attn_weights = self.encoder_mod_forward(x, mod, src_mask, src_key_padding_mask, i==0, need_weights=True)
                all_weights.append(attn_weights)
            else:
                x = self.encoder_mod_forward(x, mod, src_mask, src_key_padding_mask, i==0)

        if self.transformer.encoder.norm is not None:
            x = self.transformer.encoder.norm(x)

        if need_weights:
            return x, torch.cat(all_weights)
        else:
            return x

    def decoder_layer_forward_with_attention(self, mod, x, memory, tgt_mask=None, tgt_key_padding_mask=None, 
                                            mem_mask=None, memory_key_padding_mask=None, is_first_dec_mod=False):
        #mod_is_relative_self = isinstance(mod.self_attn, MultiHeadRelativeAttention)
        use_pe = self.positional_encoding_type=="absolute"
        def _sa_block(x, attn_mask, key_padding_mask):
            q, k, v = self.prepare_embedding_for_attn(x, is_first_dec_mod, use_pe)
            x, attn_weights = mod.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
            return mod.dropout1(x), attn_weights

        def _mha_block(x, mem, attn_mask, key_padding_mask):
            if self.positional_encoding_type=='relative-transfxl':
                if self.extra_positional_encoding_relative_decoder_mha:
                    offset = self.positional_encoding.pe.size(0) // 2 + 1
                    q = self.positional_encoding(x, pos_offset=offset)
                    k = self.positional_encoding(mem, pos_offset=offset)
                    if not self.positional_encoding_query_key_only:
                        v = self.positional_encoding(mem, pos_offset=offset)
                    else:
                        v = mem
                else:
                    q = x
                    k = mem
                    v = mem
            elif self.positional_encoding_type=='absolute':
                if self.repeat_positional_encoding:
                    q = self.positional_encoding(x)
                    # q = x
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
        tgt = self.tgt_embedding(tgt)
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