import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GINConv,GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool,global_max_pool
import matplotlib.pyplot as plt
import numpy as np

class GNN_mean(torch.nn.Module):
    def __init__(self, dim_h, num_layers, num_node_features, num_classes):
        super(GNN_mean, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        if(num_layers > 1):
            self.layers.append(GCNConv(num_node_features, dim_h))
            for i in range(1,num_layers):
                self.layers.append(GCNConv(dim_h, dim_h))
        else:
            self.layers.append(GCNConv(num_node_features, num_classes))
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, edge_index, batch,num_layers):
        
        h = x
        for i in range(num_layers):
            h = (self.layers[i](h, edge_index)).relu()
                

        hG = global_mean_pool(h, batch)

        h = F.dropout(hG, p=0.5, training=self.training)
        
        if(num_layers > 1):
            h = self.lin(h)
        
        return hG, F.log_softmax(h, dim=1)


class GNN_max(torch.nn.Module):
    def __init__(self, dim_h, num_layers, num_node_features, num_classes):
        super(GNN_max, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        if(num_layers > 1):
            self.layers.append(GCNConv(num_node_features, dim_h))
            for i in range(1,num_layers):
                self.layers.append(GCNConv(dim_h, dim_h))
    
        else:
            self.layers.append(GCNConv(num_node_features, num_classes))
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, edge_index, batch,num_layers):
        
        h = x
        for i in range(num_layers):
            h = (self.layers[i](h, edge_index)).relu()
                

        hG = global_max_pool(h, batch)

        h = F.dropout(hG, p=0.5, training=self.training)
        
        if(num_layers > 1):
            h = self.lin(h)
        
        return hG, F.log_softmax(h, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, dim_h, num_layers, num_node_features, num_classes):
        super(GIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        self.layers.append(GINConv(Sequential(Linear(num_node_features, dim_h),
                           BatchNorm1d(dim_h), ReLU(),
                           Linear(dim_h, dim_h), ReLU())))
        for i in range(1,num_layers):
            self.layers.append(GINConv(Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                                   Linear(dim_h, dim_h), ReLU())))
        self.fin = Linear(dim_h*num_layers, dim_h*num_layers)
        self.out = Linear(dim_h*num_layers, num_classes)
            


    def forward(self, x, edge_index, batch, num_layers):
        h = x
        for i in range(num_layers):
            h = self.layers[i](h, edge_index) 
            h_sum = global_add_pool(h, batch)
            if(i == 0):
                h_fin = h_sum
            else:
                h_fin = torch.cat((h_fin,h_sum), dim=1)

        
        h_fin = (self.fin(h_fin)).relu()
        h_fin = F.dropout(h_fin, p=0.5, training=self.training)
        h_fin = self.out(h_fin)
        
        return h_fin, F.log_softmax(h_fin, dim=1)
    

def train(model, loader,num_layers, test_loader):
    test_fin = []
    train_fin = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=0.01)
    epochs = 100
    train_acc=[]
    model.train()
    for epoch in range(epochs+1):
        acc = 0
        tot = 0

        for data in loader:
            optimizer.zero_grad()
            _, out = model(data.x, data.edge_index, data.batch, num_layers)
            loss = criterion(out, data.y)
            acc += (out.argmax(dim=1) == data.y).sum()
            loss.backward()
            optimizer.step()
            tot = tot + len(data.y)
        train_acc.append(acc/(tot))

    print((np.mean(train_acc)).item())
    train_fin.append((np.mean(train_acc)).item())
    test_acc = test(model, test_loader,num_layers)
    print(test_acc.item())
    test_fin.append(test_acc.item())
    
    return model,train_fin,test_fin

@torch.no_grad()
def test(model, loader, num_layers):
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    acc = 0
    tot = 0

    for data in loader:
        _, out = model(data.x, data.edge_index, data.batch, num_layers)
        acc += (out.argmax(dim=1) == data.y).sum()
        tot = tot+len(data.y)
    acc = acc/tot
    
    return acc
