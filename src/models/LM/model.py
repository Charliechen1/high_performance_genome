import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Sublayer import MultiHeadAttention, PositionwiseFeedForward

class LSTMTagger(nn.Module):

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
                 need_attn=True,
                 dropout=0.1):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_k = d_k
        self.n_head = n_head
        self.need_attn = need_attn
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=n_lstm)
        if need_attn:
            self.hid2hid = nn.Linear(seq_len, 1)
        else:
            self.hid2hid = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        self.slf_attn = MultiHeadAttention(n_head, hidden_dim * 2, seq_len, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(hidden_dim * 2, hidden_dim * 2, seq_len, dropout=dropout)
        
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.dropout = nn.Dropout(p=dropout)
        self.padding_idx = padding_idx
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X, slf_attn_mask=None):
        b_size, seq_len = X.size()
        embeds = self.word_embeddings(X)
        lstm_hid, _ = self.lstm(embeds.view(b_size, seq_len, -1))
        lstm_hid = lstm_hid.transpose(1, 2)
        
        # take self-attention on the hidden states of the LSTM model
        # start attention layer
        
        ###### self attention version ######
        if self.need_attn:
            enc_out, enc_slf_attn = self.slf_attn(
                lstm_hid, lstm_hid, lstm_hid, mask=slf_attn_mask)
            attn_out = self.pos_ffn(enc_out)
        ####################################
        else:
            # default, take the last layer as output of LSTM
            #attn_out = lstm_hid[:, :, -1]
            attn_out = lstm_hid.mean(2)
        # linear transformation and classification layer
        #hidden_res = self.relu(self.hid2hid(attn_out))
        hidden_res = attn_out.mean(2)
        hidden_res = hidden_res.view(b_size, self.hidden_dim * 2)
        tag_space = self.hidden2tag(hidden_res)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores