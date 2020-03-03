import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, seq_len, vocab_size, tagset_size, padding_idx=0, d_k=64, num_of_header=2):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_k = d_k
        self.num_of_header = num_of_header
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.hid2hid = nn.Linear(hidden_dim * 2 * num_of_header, hidden_dim * 2)
        
        # attention related layers
        self.Wq = nn.Linear(seq_len, num_of_header)
        self.Wk = nn.Linear(seq_len, num_of_header)
        self.Wv = nn.Linear(seq_len, num_of_header)
        
        #self.attn = nn.Linear(seq_len, 1)
        
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.dropout_layer = nn.Dropout(p=0.75)
        self.padding_idx = padding_idx
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X):
        X_size, seq_len = X.size()
        embeds = self.word_embeddings(X)
        lstm_hid, _ = self.lstm(embeds.view(X_size, seq_len, -1))
        lstm_hid = lstm_hid.transpose(1, 2)
        
        # take self-attention on the hidden states of the LSTM model
        # start attention layer
        
        ###### most trivial version ########
        # attn_out = lstm_hid.mean(1)
        ####################################
        
        ###### naive attention version #####
        # attn_out = self.attn(lstm_hid)
        # attn_out = attn_out.view(X_size, self.hidden_dim * 2)
        ####################################
        
        ###### self attention version ######
        
        Q = self.Wq(lstm_hid).view(X_size, self.hidden_dim * 2 * self.num_of_header)
        K = self.Wq(lstm_hid).view(X_size, self.hidden_dim * 2 * self.num_of_header)
        V = self.Wq(lstm_hid).view(X_size, self.hidden_dim * 2 * self.num_of_header)
        
        K_norm = F.normalize(K, p=2, dim=1)
        #K_norm = K / torch.sqrt(torch.Tensor(self.d_k))
        attn = torch.matmul(Q, K_norm.transpose(0, 1))
        attn_out = torch.matmul(attn, V)
        ####################################
        
        # linear transformation and classification layer
        hidden_res = self.relu(self.hid2hid(attn_out))
        tag_space = self.hidden2tag(hidden_res)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores