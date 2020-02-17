import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, padding_idx=0):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hid2hid1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.hid2hid2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout_layer = nn.Dropout(p=0.75)
        self.padding_idx = padding_idx
        self.softmax = nn.LogSoftmax()
    
    def forward(self, X):
        X_size, seq_len = X.size()
        embeds = self.word_embeddings(X)
        lstm_out, _ = self.lstm(embeds.view(X_size, seq_len, -1))
        lstm_out = lstm_out.view(X_size, seq_len, -1)
        # current we just take the last hidden state of the LSTM, later will modify to attention layer
        # we do not want to take the state for padding
        last_state = lstm_out.mean(1)
        hidden_res = F.relu(self.hid2hid1(last_state))
        hidden_res = F.relu(self.hid2hid2(hidden_res))
        tag_space = self.hidden2tag(hidden_res)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores