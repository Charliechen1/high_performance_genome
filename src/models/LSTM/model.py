import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size, padding_idx=0):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.batch_size = batch_size
        self.padding_idx = padding_idx
        self.softmax = nn.LogSoftmax()
    
    def forward(self, X):
        batch_size, seq_len = X.size()
        embeds = self.word_embeddings(X)
        lstm_out, _ = self.lstm(embeds.view(batch_size, seq_len, -1))
        lstm_out = lstm_out.view(batch_size, seq_len, -1)
        # current we just take the last hidden state of the LSTM, later will modify to attention layer
        # we do not want to take the state for padding
        last_state = lstm_out.mean(1)
        tag_space = self.hidden2tag(last_state)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores