import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementations based on attention in pytorch (normal attention)
# And https://github.com/RobertCsordas/transformer_generalization (relative attention)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)

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
        # if self.bias:
        #     nn.init.constant_(self.w_q.bias, 0.)
        #     nn.init.constant_(self.w_k.bias, 0.)
        #     nn.init.constant_(self.w_v.bias, 0.)
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

        q = q * self.scaling
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

# Adapted from https://github.com/RobertCsordas/transformer_generalization/blob/38f2734f5dcad331d4a9658b73b33889ce607c87/layers/transformer/multi_head_relative_pos_attention.py#L51
class MultiHeadRelativeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, pe_mod, dropout=0., bias=True):
        super(MultiHeadRelativeAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.pe_mod = pe_mod
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k_pos = nn.Linear(embed_dim, embed_dim, bias=False)
        self.global_content_bias = nn.Parameter(torch.zeros([self.num_heads, self.head_dim]))
        self.global_pos_bias = nn.Parameter(torch.zeros([self.num_heads, self.head_dim]))

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
        nn.init.xavier_uniform_(self.w_k_pos.weight, 1/2)
        nn.init.constant_(self.global_content_bias, 0.)
        nn.init.constant_(self.global_pos_bias, 0.)
        # nn.init.xavier_uniform_(self.out_proj.weight)
        # if self.bias:
        #     nn.init.constant_(self.w_q.bias, 0.)
        #     nn.init.constant_(self.w_k.bias, 0.)
        #     nn.init.constant_(self.w_v.bias, 0.)
            # nn.init.constant_(_self.w_k_pos.bias, 0.)
            # nn.init.constant_(self.out_proj.bias, 0.)

    def get_pos_enc(self, length, offset):
        size = 2 * length - 1
        pe_weight = self.pe_mod.pe
        lower = pe_weight.size(0)//2 - length + 1 - offset
        return pe_weight.narrow(0, lower, size)

    def add_head_specific_bias(self, data, bias):
        return (data.view(-1, bias.shape[0], *data.shape[1:]) + bias.unsqueeze(1).type_as(data)).view_as(data)

    def _shift(self, posmat: torch.Tensor) -> torch.Tensor:
        # Slice out a matrix diagonally. Each successive row is sliced one position to the left compared.
        # shape: [n_batch, n_head, n_out, n_in * 2 - 1]
        # return: [n_batch, n_head, n_out, n_in]
        p = F.pad(posmat, (0, 1, 0, 1)).flatten(-2)  # [n_batch, n_head, (n_out + 1) * n_in * 2]
        p = p.narrow(-1, posmat.shape[-1] // 2, posmat.shape[-1] * posmat.shape[-2]).view_as(posmat)

        return p.narrow(-1, 0, (posmat.shape[-1] + 1) // 2)


    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        # Inputs: q, k, v
        query_len, batch_size, embed_dim = query.size()
        key_len, _, _ = key.size()

        # Q = q * wQ, k = k * wK, v = v*wV
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(query_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(key_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(value.size(0), batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # n_heads, 2*key_len - 1, head_dim
        # 2nd dimension (2*key_len-1) corresponds to all possible differences between tokens (first - last, ... last - first)
        # Also goes through it's own separate weight matrix (wR in the paper)
        k_pos = self.w_k_pos(self.get_pos_enc(key_len, 0)).view(-1, self.num_heads, self.head_dim).transpose(0,1)

        # q_content = (q + u), q_pos = (q + v) (same u, v in paper)
        q_content = self.add_head_specific_bias(q, self.global_content_bias)
        q_pos = self.add_head_specific_bias(q, self.global_pos_bias)

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

        # q_content * K.T = (q + u) * k.T = q * k.T + u * k.T = terms a, c in paper
        content_attn = torch.bmm(q_content, k.transpose(1, 2))

        # pos_attn size: [n_batch, n_head, n_out, n_in * 2 - 1]
        pos_attn = torch.matmul(q_pos.view(batch_size, self.num_heads, query_len, -1), k_pos.transpose(-1, -2))
        # pos_attn_size: [n_batch * n_head, query_len, key_len]
        pos_attn = self._shift(pos_attn).flatten(0, 1)

        attn = content_attn + pos_attn
        attn = attn * self.scaling

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