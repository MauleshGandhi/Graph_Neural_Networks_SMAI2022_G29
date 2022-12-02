import torch
from torch_geometric.datasets import TUDataset
from test_dataset import test_datasets

dataset_name = ['MUTAG','NCI1', 'PROTEINS']
for name in dataset_name:
    dataset = TUDataset(root='.', name=name).shuffle()
    test_datasets(dataset,name)
