from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from Train_Test import train,test
from Train_Test import GNN_mean,GNN_max,GIN

def test_datasets(dataset, name):
    train_dataset = dataset[:int(len(dataset)*0.8)]
    test_dataset  = dataset[int(len(dataset)*0.8):]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    test_fin = []
    train_fin = []
    
    num_layers = [1,3]
    
    for num in num_layers:
        
        gnn_max = GNN_max(dim_h=32,num_layers=num, num_node_features = dataset.num_node_features,num_classes = dataset.num_classes)
        gnn_mean = GNN_mean(dim_h=32,num_layers=num, num_node_features = dataset.num_node_features,num_classes = dataset.num_classes)
        gin = GIN(dim_h=32, num_layers=num, num_node_features = dataset.num_node_features,num_classes = dataset.num_classes)

        gnn_max,train1,test1 = train(gnn_max, train_loader, num,test_loader)
        gnn_mean,train2,test2 = train(gnn_mean, train_loader, num, test_loader)
        gin,train3,test3 = train(gin, train_loader, num, test_loader)
    
        train_fin.append(train1)
        train_fin.append(train2)
        train_fin.append(train3)
        test_fin.append(test1)
        test_fin.append(test2)
        test_fin.append(test3)


    fig = plt.figure(figsize=(15,5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 

    ax.plot(train_fin)
    ax.plot(test_fin)

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title(name)
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['Max-1-Layer','Mean-1-Layer','Sum-1-Layer','Max-MLP','Mean-MLP','Sum-MLP'])
    ax.legend(["Training", "Testing"], loc ="lower right")

    plt.show()
