import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool as gap

import sys
sys.path.append('../LM')
from Sublayer import PositionalEncoding



class GCN(torch.nn.Module):
    def __init__(self, target_dim, lm):
        super(GCN, self).__init__()
        self.lm = lm
        
        if lm is None:
            
            self.embedding = torch.nn.Embedding(27, 80)
            self.conv1 = GCNConv(80, 128)
            
        else:
        
            lm_embedding = lm['module.word_embeddings.weight']
            self.embedding = torch.nn.Embedding.from_pretrained(lm_embedding, freeze = True)            
            self.conv1 = GCNConv(80, 128)           
            
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 256)
        self.linear1 = torch.nn.Linear(256, 512)
        self.linear2 = torch.nn.Linear(512, target_dim)

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
                
        #if self.lm is None:
            
        #    x = F.one_hot(x.long(), num_classes = 27).float()
            
        #else:
            
        x = self.embedding(x.long())
        
        #if self.n_lstm > 0:
        #    
        #    x, _ = self.lstm(x.unsqueeze(0))
        #    x = x.squeeze(0)
            
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p =.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = .1, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = .1, training=self.training)
        x = gap(x, batch)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p = .1, training=self.training)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        
        return x
    
class Toy(torch.nn.Module):
    
    def __init__(self, target_dim, lm):
        
        super(Toy, self).__init__()
        self.lm = lm
        
        if lm is None:
            
            self.embedding = torch.nn.Embedding(27, 80)
            
        else:
        
            lm_embedding = lm['module.word_embeddings.weight']
            self.embedding = torch.nn.Embedding.from_pretrained(lm_embedding, freeze = True)   
            
        self.linear1 = torch.nn.Linear(80, 128)
        self.linear2 = torch.nn.Linear(128, target_dim)
        
    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.embedding(x.long())
        x = gap(x, batch)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p = .1, training=self.training)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        
        return x