import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Sublayer import MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, PositionalEncoding

class LSTMAttn(nn.Module):

    def __init__(self, embedding_dim, 
                 hidden_dim, 
                 seq_len, 
                 vocab_size, 
                 tagset_size, 
                 padding_idx=0, 
                 d_k=64, 
                 d_v=64,
                 n_lstm=3, 
                 n_head=12,
                 n_attn=2,
                 dropout=0.1,
                 need_pos_enc=True):
        super(LSTMAttn, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_k = d_k
        self.n_head = n_head
        self.n_attn = n_attn
        self.n_lstm = n_lstm
        self.need_pos_enc = need_pos_enc
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        if need_pos_enc:
            self.position_enc = PositionalEncoding(embedding_dim, n_position=seq_len)
        
        if n_lstm:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=n_lstm)
        else:
            # if we decided not to use a LSTM there, means there will be no "hidden dimension", 
            # and simply we use embedding dimension for the further part.
            # we treat LSTM as transparent.
            # Divide 2 because the bi-LSTM actually output hidden_dim * 2 
            self.hidden_dim = embedding_dim // 2
        if n_attn:
            self.hid2hid = nn.Linear(seq_len, 1)
        else:
            self.hid2hid = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        
        self.slf_attn_encoder = EncoderLayer(n_head, self.hidden_dim, seq_len, d_k, d_v, dropout=dropout)
        
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, tagset_size)
        self.dropout = nn.Dropout(p=dropout)
        self.padding_idx = padding_idx
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X, slf_attn_mask=None):
        b_size, seq_len = X.size()
        
        #embedding + LSTM part
        enc_output = self.word_embeddings(X)
        if self.need_pos_enc:
            enc_output = self.position_enc(enc_output)
        enc_output = self.dropout(enc_output)
        if self.n_lstm:
            # here "if" is for flexibility of taking LSTM or not.
            enc_output, _ = self.lstm(enc_output.view(b_size, seq_len, -1))
        
        enc_output = enc_output.transpose(1, 2)
        
        # take self-attention on the hidden states of the LSTM model
        # start attention layer
        
        ###### self attention version ######
        if self.n_attn:
            attn_out = enc_output
            for _ in range(self.n_attn):
                attn_out, slf_attn_res = self.slf_attn_encoder(attn_out)
        ####################################
        else:
            # default, take the last layer as output of LSTM
            attn_out = embeds.mean(2)
        # linear transformation and classification layer
        hidden_res = attn_out.mean(2)
        hidden_res = hidden_res.view(b_size, self.hidden_dim * 2)
        tag_space = self.hidden2tag(hidden_res)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    