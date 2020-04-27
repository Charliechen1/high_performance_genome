import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool as gap

class GCN(torch.nn.Module):
    def __init__(self, target_dim, lm_embedding):
        super(GCN, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(lm_embedding, freeze = True)
        #self.embedding = torch.nn.Embedding(27, 80)
        self.conv1 = GCNConv(80, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 256)
        self.linear1 = torch.nn.Linear(256, 512)
        self.linear2 = torch.nn.Linear(512, target_dim)

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
                                
        x = self.embedding(x.long())
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = gap(x, batch)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        
        return x
    
# class Toy(torch.nn.Module):
class LinearClassifier(torch.nn.Module):
    
    def __init__(self, target_dim, lm_embedding):
        
        super(Toy, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(lm_embedding, freeze = True)
        self.linear1 = torch.nn.Linear(80, 128)
        self.linear2 = torch.nn.Linear(128, target_dim)
        
    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.embedding(x.long())
        x = gap(x, batch)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        
        return x