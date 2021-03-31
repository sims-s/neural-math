import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

class PositionalEncoding(nn.Module):
    # From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerEmbedding(nn.Module):
    def __init__(self, n_tokens, embed_dim, max_decode_size, dropout, scale_embeddings):
        super(TransformerEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_tokens, embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout, max_decode_size)
        self.scale_embeddings = scale_embeddings
        
    def forward(self, x):
        if self.scale_embeddings:
            x = np.sqrt(self.embed_dim) * self.embedding(x).transpose(0,1)
        else:
            x = self.embedding(x).transpose(0,1)
        x = self.pe(x)
        return x


class Factorizer(nn.Module):
    def __init__(self, n_tokens, embed_dim, max_decode_size, shared_embeddings, pad_token_id, scale_embeddings, **kwargs):
        super(Factorizer, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.pad_token_id = pad_token_id

        dropout = .1 if not 'dropout' in kwargs else kwargs['dropout']
        if not shared_embeddings:
            self.src_embedding = TransformerEmbedding(n_tokens, embed_dim, max_decode_size, dropout, scale_embeddings)
            self.tgt_embedding = TransformerEmbedding(n_tokens, embed_dim, max_decode_size, dropout, scale_embeddings)
        else:
            self.embedding = TransformerEmbedding(n_tokens, embed_dim, max_decode_size, dropout, scale_embeddings)
        
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
    
    def decode(self, tgt, memory, memory_key_padding_mask):
        tgt_key_padding_mask = self.form_pad_mask(tgt)
        tgt = self.route_embeddings(tgt, 'tgt')
        
        tgt_mask = self.form_subsequence_mask(tgt)
        
        output = self.transformer.decoder(tgt, memory, tgt_mask = tgt_mask, 
                                          tgt_key_padding_mask = tgt_key_padding_mask,
                                          memory_key_padding_mask = memory_key_padding_mask)
        output = output.transpose(0,1)
        decoded = self.tokens_out(output)
        return decoded
    
        
    def forward(self, src, tgt):
        memory, memory_key_padding_mask = self.encode(src)
        decoded = self.decode(tgt, memory, memory_key_padding_mask)
        return decoded