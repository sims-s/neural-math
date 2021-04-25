import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

class PositionalEncoding(nn.Module):
    # From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout, max_len, learnable):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        if learnable:
            self.pe = nn.Parameter(pe)
        else:
            self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerEmbedding(nn.Module):
    def __init__(self, n_tokens, embed_dim, max_decode_size, dropout, scale_embeddings, learn_positional_encoding):
        super(TransformerEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_tokens, embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout, max_decode_size, learn_positional_encoding)
        self.scale_embeddings = scale_embeddings
        
    def forward(self, x):
        if self.scale_embeddings:
            x = np.sqrt(self.embed_dim) * self.embedding(x).transpose(0,1)
        else:
            x = self.embedding(x).transpose(0,1)
        x = self.pe(x)
        return x


class Factorizer(nn.Module):
    def __init__(self, n_tokens, embed_dim, max_decode_size, shared_embeddings, pad_token_id, scale_embeddings, learn_positional_encoding, **kwargs):
        super(Factorizer, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.pad_token_id = pad_token_id

        dropout = .1 if not 'dropout' in kwargs else kwargs['dropout']
        if not shared_embeddings:
            self.src_embedding = TransformerEmbedding(n_tokens, embed_dim, max_decode_size, dropout, scale_embeddings, learn_positional_encoding)
            self.tgt_embedding = TransformerEmbedding(n_tokens, embed_dim, max_decode_size, dropout, scale_embeddings, learn_positional_encoding)
        else:
            self.embedding = TransformerEmbedding(n_tokens, embed_dim, max_decode_size, dropout, scale_embeddings, learn_positional_encoding)
        
        self.transformer = nn.Transformer(d_model = embed_dim, **kwargs)
        
        self.tokens_out = nn.Linear(embed_dim, n_tokens)
        
    def form_pad_mask(self, tokens):
        return (tokens==self.pad_token_id).to(tokens.device)
    
    def form_subsequence_mask(self, tgt):
        size = tgt.size(0)
        return (torch.triu(torch.ones(size, size)) == 0).transpose(0,1).to(tgt.device)
    
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
        
        return self.transformer.encoder(src, src_key_padding_mask = src_key_padding_mask), src_key_padding_mask

    def decoder_layer_forward_with_attention(self, layer, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        tgt2 = layer.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + layer.dropout(tgt2)
        tgt = layer.norm1(tgt)
        tgt2, attn = layer.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + layer.dropout2(tgt2)
        tgt = layer.norm2(tgt)
        tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt))))
        tgt = tgt + layer.dropout3(tgt2)
        tgt = layer.norm3(tgt)
        return tgt, attn


    def decoder_forward_with_attention(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        output = tgt
        attn_list = []
        for mod in self.transformer.decoder.layers:
            output, attn = self.decoder_layer_forward_with_attention(mod, output, memory, tgt_mask=tgt_mask, 
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
            attn_list.append(attn)
        if self.transformer.decoder.norm is not None:
            output = self.transformer.decoder.norm(output)
        
        return output, attn_list

    def decode(self, tgt, memory, memory_key_padding_mask, return_enc_dec_attn=False):
        tgt_key_padding_mask = self.form_pad_mask(tgt)
        tgt = self.route_embeddings(tgt, 'tgt')
        
        tgt_mask = self.form_subsequence_mask(tgt)
        
        if not return_enc_dec_attn:
            output = self.transformer.decoder(tgt, memory, tgt_mask = tgt_mask, 
                                          tgt_key_padding_mask = tgt_key_padding_mask,
                                          memory_key_padding_mask = memory_key_padding_mask)
        else:
            output, attn_weights = self.decoder_forward_with_attention(tgt, memory, tgt_mask=tgt_mask,
                                    tgt_key_padding_mask = tgt_key_padding_mask, 
                                    memory_key_padding_mask = memory_key_padding_mask)
        output = output.transpose(0,1)
        decoded = self.tokens_out(output)
        if not return_enc_dec_attn:
            return decoded
        else:
            return decoded, attn_weights
    
        
    def forward(self, src, tgt):
        memory, memory_key_padding_mask = self.encode(src)
        decoded = self.decode(tgt, memory, memory_key_padding_mask)
        return decoded