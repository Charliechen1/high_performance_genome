import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearClassifier(nn.Module):
    ''' 
    The linear classifier
    '''

    def __init__(self, hidden_dim, tagset_size):
        """
        Linear Classifier for pFam family
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        b_size = x.shape[0]
        hidden_res = x.mean(2)
        hidden_res = hidden_res.view(b_size, self.hidden_dim * 2)
        tag_space = self.hidden2tag(hidden_res)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
class MaskedLanguageModel(nn.Module):
    
    def __init__(self, hidden_dim, vocab_size):
        """
        the masked language model
        """
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        runner = x.transpose(1, 2)
        runner = self.softmax(self.linear(runner))
        runner = runner.transpose(2, 1)
        return runner