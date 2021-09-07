import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.scaling = (self.head_dim) ** -0.5

        self._reset_parameters()

    def _reset_parameters(self):
        # Pytorch implementation concatenates w_q, w_k, w_v. If D is embedding dimension, 
        # PT implementation has fan_in=3D, but my implementation would have fan_in=D for each layer
        # Setting the gain to 1/2 makes these initializations equivelent
        # For details, see: https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform
        nn.init.xavier_uniform_(self.w_q.weight, 1/2)
        nn.init.xavier_uniform_(self.w_k.weight, 1/2)
        nn.init.xavier_uniform_(self.w_v.weight, 1/2)
        # nn.init.xavier_uniform_(self.out_proj.weight)
        if self.bias:
            nn.init.constant_(self.w_q.bias, 0.)
            nn.init.constant_(self.w_k.bias, 0.)
            nn.init.constant_(self.w_v.bias, 0.)
            # nn.init.constant_(self.out_proj.bias, 0)


    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        query_len, batch_size, embed_dim = query.size()
        key_len, _, _ = key.size()

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(query_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(key_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(value.size(0), batch_size * self.num_heads, self.head_dim).transpose(0, 1)


        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, key_len).expand(-1, self.num_heads, -1, -1)
            key_padding_mask = key_padding_mask.reshape(batch_size * self.num_heads, 1, key_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else: 
                attn_mask = attn_mask.logical_or(key_padding_mask)
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))

        q = q / math.sqrt(q.size(2))
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += new_attn_mask
        attn_weight = torch.softmax(attn, dim=-1)
        if self.training:
            attn_weight = F.dropout(attn_weight, self.dropout_p)
        output = torch.bmm(attn_weight, v)
        output = output.transpose(0, 1).contiguous().view(query_len, batch_size, embed_dim)
        output = self.out_proj(output)

        if need_weights:
            attn_weight = attn_weight.view(batch_size, self.num_heads, query_len, key_len)
            return output, attn_weight
        else:
            return output