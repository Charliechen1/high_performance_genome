import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
Modified from https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head, hidden_dim, seq_len, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, hidden_dim * 2, seq_len, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(hidden_dim * 2, hidden_dim * 2, seq_len, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    """
    The multi head attention layer
    """
    def __init__(self, n_head, d_model, seq_len, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(seq_len, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(seq_len, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(seq_len, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, seq_len, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([d_model, seq_len], eps=1e-6)

        
    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        q = self.layer_norm(q)
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """
    The feed forward part
    """
    def __init__(self, d_in, d_hid, seq_len, dropout=0.5):
        super().__init__()
        self.w_1 = nn.Linear(seq_len, seq_len) # position-wise
        self.w_2 = nn.Linear(seq_len, seq_len) # position-wise
        self.layer_norm = nn.LayerNorm([d_in, seq_len], eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x
    
class ScaledDotProductAttention(nn.Module):
    """
    Dot product and softmax of Q K and V
    """
    def __init__(self, temperature, attn_dropout=0.5):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        k_norm = F.normalize(k, p=2, dim=1)
        attn = torch.matmul(q, k_norm.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn