
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    BatchNorm1d,
    Linear,
    ModuleList,
    ReLU,
    Sequential
)

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GINEConv, GPSConv, GATConv, GatedGraphConv, global_mean_pool
from torch_geometric.nn import PositionalEncoding
from torch_geometric.nn.attention import PerformerAttention

from sklearn.model_selection import train_test_split

def save_state(model, name : str):
    state_dict = model.state_dict()
    torch.save(state_dict, f'{name}_state.pth')

class mlp_embedding(torch.nn.Module):
    def __init__(self, dim_in:int, dim_out: int):
        super().__init__()
        
        self.embed = nn.Sequential(nn.Linear(dim_in, 2 * dim_out),
                                   nn.ReLU(),
                                   nn.Linear(2 * dim_out, 2 * dim_out),
                                   nn.ReLU(),
                                   nn.Linear(2 * dim_out, dim_out))
        
    def forward(self, x):
        embedding = self.embed(x)
        return embedding

class GCNRegression(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, dropout_p):
        super(GCNRegression, self).__init__()
        torch.manual_seed(12345)
        self.layers = num_layers
        self.dropout = dropout_p
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)  

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        for i in range(self.layers):
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        x = global_mean_pool(x, batch) 
        x = self.fc(x) 
        return x
    
class GPS(nn.Module):
    def __init__(self, channels: int, out_channels: int, num_node_feats: int, num_edge_feats: int, mpnn_type: str, num_heads: int, num_layers: int, attn_type: str, pe_dim: int, task: str='regression'):
        super().__init__()

        self.task = task
        self.mpnn_type = mpnn_type
        self.num_node_feats = num_node_feats

        self.projection = nn.Linear(2*channels, channels)
        self.node_embedding = mlp_embedding(num_node_feats, channels) #change to embedding
        self.edge_embedding = mlp_embedding(num_edge_feats, channels)

        self.pe_norm = nn.BatchNorm1d(pe_dim)
        self.pe_lin = nn.Linear(pe_dim, channels)

        neuralnet = Sequential(nn.Linear(channels, channels),
                               nn.ReLU(),
                               nn.Linear(channels, channels))

        if mpnn_type == "GCN":
            message_passing = GCNConv(channels, channels)
        elif mpnn_type == "GatedGCN":
            message_passing = GatedGraphConv(channels, 4)
        elif mpnn_type == "GINE":
            message_passing = GINEConv(neuralnet, edge_dim=channels)
        elif mpnn_type == "GAT":
            message_passing = GATConv(channels, channels) #heads=num_heads
        else:
            raise Exception("MPNN type not recognized")

        self.convs = ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GPSConv(channels=channels, 
                        conv=message_passing, 
                        heads=num_heads, 
                        attn_type=attn_type))

        if task == 'regression':
            self.prediction_head = nn.Sequential(nn.Linear(channels, channels // 2),
                                                 nn.ReLU(),
                                                 nn.Linear(channels // 2, channels // 4),
                                                 nn.ReLU(),
                                                 nn.Linear(channels // 4, out_channels))
            self.criterion = nn.MSELoss()
        elif task == 'classification':
            self.prediction_head = nn.Linear(channels, out_channels)
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise Exception("Task must be either regression or classification!")

    def forward(self, data):
        x, pe, edge_index, edge_attr, batch = data.x, data.pe, data.edge_index, data.edge_attr, data.batch

        x_pe = self.pe_norm(pe)
        x_pe_lin = self.pe_lin(x_pe)

        node_emb = self.node_embedding(x)
        
        x = torch.cat((node_emb, x_pe_lin), -1)
        x = self.projection(x)
        
        edge_attr = self.edge_embedding(edge_attr)

        for conv in self.convs: 
            if self.mpnn_type in ["GCN", "GatedGCN"]:
                x = conv(x, edge_index, batch)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)
        out = self.prediction_head(x)
        return out

    def compute_loss(self, preds, targets):
        return self.criterion(preds, targets)

    def prepare_targets(self, batch, device):
        if self.task == 'regression':
            return batch.y.squeeze().to(device)
        elif self.task == 'classification':
            if len(batch.y.shape) > 1:
                # If batch.y is one-hot encoded or has extra dimensions
                if batch.y.shape[1] > 1:
                    # Convert one-hot encoding to class indices
                    return batch.y.argmax(dim=1).to(device).long()
                else:
                    # Just remove the extra dimension
                    return batch.y.squeeze().to(device).long()
            else:
                # Already correctly shaped
                return batch.y.to(device).long()
        else:
            raise ValueError("Task must be 'regression' or 'classification'")
    

