import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import time

from sklearn.utils import shuffle
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt

from helpers import GNN, plots, dataLoader

dtypeFloat = torch.cuda.FloatTensor
dtypeLong = torch.cuda.LongTensor

num_nodes = 49
num_neighbors = -1    # when set to -1, it considers all the connections instead of k nearest neighbors
train_filepath = f"./dataset/CVRP{num_nodes}_train_GNN.txt"

dataset = dataLoader.GoogleCVRPReader(num_nodes, num_neighbors, 1, train_filepath)

t = time.time()
batch = next(iter(dataset))  # Generate a batch of CVRP
print("Batch generation took: {:.3f} sec".format(time.time() - t))

print("edges:", batch.edges.shape)
print("edges_values:", batch.edges_values.shape)
print("edges_targets:", batch.edges_target.shape)
print("nodes:", batch.nodes.shape)
print("nodes_target:", batch.nodes_target.shape)
print("nodes_coord:", batch.nodes_coord.shape)
print("nodes_features:", batch.nodes_features.shape)
print("tour_nodes:", batch.tour_nodes.shape)
print("tour_len:", batch.tour_len.shape)

#idx = 0
#f = plt.figure(figsize=(5, 5))
#a = f.add_subplot(111)
#plots.plot_cvrp(a, batch.nodes_coord[idx], batch.edges[idx], batch.edges_values[idx], batch.edges_target[idx])

for x in [3]:

    num_neighbors = -1 # Could increase it!
    train_filepath = f"./dataset/CVRP{num_nodes}_train_GNN.txt"
    hidden_dim = 100*x #@param
    num_layers = 10*x #@param
    mlp_layers = 3 #@param
    learning_rate = 0.001 #@param
    max_epochs = 700 #@param
    batches_per_epoch = 800

    variables = {'train_filepath': f'./dataset/CVRP{num_nodes}_train_GNN.txt', 
                'val_filepath': f'./dataset/CVRP{num_nodes}_val_GNN.txt', 
                'test_filepath': f'./dataset/CVRP{num_nodes}_test_GNN.txt', 
                'num_nodes': num_nodes, 
                'num_neighbors': num_neighbors, 
                'node_dim': 2 , 
                'voc_nodes_in': 2, 
                'voc_nodes_out': 2, 
                'voc_edges_in': 3, 
                'voc_edges_out': 2, 
                'hidden_dim': hidden_dim, 
                'num_layers': num_layers, 
                'mlp_layers': mlp_layers, 
                'aggregation': 'mean', 
                'max_epochs': max_epochs, 
                'val_every': 5, 
                'test_every': 5, 
                'batches_per_epoch': batches_per_epoch, 
                'accumulation_steps': 1, 
                'learning_rate': learning_rate, 
                'decay_rate': 1.01
                }
    net = nn.DataParallel(GNN.ResidualGatedGCNModel(variables,  torch.cuda.FloatTensor, torch.cuda.LongTensor))
    net.cuda()

    # Compute number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=variables["learning_rate"])
    val_loss_old = None

    train_losses = []
    val_losses = []
    test_losses = []


    t = time.time()
    for epoch in range(variables["max_epochs"]):
        # Train
        train_time, train_loss = GNN.train_one_epoch(net, optimizer, variables)
        # Print metrics
        train_losses.append(train_loss)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}")
        
        if epoch % variables["val_every"] == 0 or epoch == variables["max_epochs"]-1:
            # Validate
            val_time, val_loss = GNN.test(net, variables, mode='val')
            val_losses.append(val_loss)
            print(f"Epoch: {epoch}, Val Loss; {val_loss}")

            # Update learning rate
            if val_loss_old != None and val_loss > 0.99 * val_loss_old:
                variables["learning_rate"] /= variables["decay_rate"]
                optimizer = GNN.update_learning_rate(optimizer, variables["learning_rate"])
            
            val_loss_old = val_loss  # Update old validation loss

        if epoch % variables["test_every"] == 0 or epoch == variables["max_epochs"]-1:
            # Test
            test_time, test_loss = GNN.test(net, variables, mode='test')
            test_losses.append(test_loss)
            print(f"Epoch: {epoch}, Test Loss; {test_loss}\n")

        # if time.time() - t >= 72000:
        #     break

        if epoch%50 == 0:
            torch.save(net, f'./model/model_{num_nodes}_{epoch}_{num_layers}_{hidden_dim}.pt')



    torch.save(net, f'./model/model_{num_nodes}_{epoch}_{num_layers}_{hidden_dim}.pt')

    with open("./plots_and_data/time_table.txt", "a") as file:
        rounded = round((time.time() - t)/epoch, 3) 
        file.write(f"Averge time for one epoch ({num_nodes}_{num_layers}_{hidden_dim}): {rounded} seconds\n")

    name = f'./plots_and_data/model_{num_nodes}_{epoch}_{num_layers}_{hidden_dim}.png'
    plots.plot_loss_curve(train_losses, val_losses, test_losses, variables, name)


# Set evaluation mode
# net.eval()

# num_samples = 10
# num_nodes = variables['num_nodes']
# num_neighbors = variables['num_neighbors']
# test_filepath = variables['test_filepath']
# dataset = iter(dataLoader.GoogleCVRPReader(num_nodes, num_neighbors, 1, test_filepath))


# x_edges = []
# x_edges_values = []
# x_nodes = []
# x_nodes_coord = []
# x_nodes_features = []
# y_edges = []
# y_nodes = []
# y_preds = []

# with torch.no_grad():
#     for i in range(num_samples):
#         sample = next(dataset)
#         # Convert batch to torch Variables
#         x_edges.append(Variable(torch.LongTensor(sample.edges).type(dtypeLong), requires_grad=False))
#         x_edges_values.append(Variable(torch.FloatTensor(sample.edges_values).type(dtypeFloat), requires_grad=False))
#         x_nodes.append(Variable(torch.LongTensor(sample.nodes).type(dtypeLong), requires_grad=False))
#         x_nodes_coord.append(Variable(torch.FloatTensor(sample.nodes_coord).type(dtypeFloat), requires_grad=False))
#         x_nodes_features.append(Variable(torch.FloatTensor(sample.nodes_features).type(dtypeFloat), requires_grad=False))
#         y_edges.append(Variable(torch.LongTensor(sample.edges_target).type(dtypeLong), requires_grad=False))
#         y_nodes.append(Variable(torch.LongTensor(sample.nodes_target).type(dtypeLong), requires_grad=False))
        
#         # Compute class weights
#         edge_labels = (y_edges[-1].cpu().numpy().flatten())
#         edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        
#         # Forward pass
#         y_pred, loss = net.forward(x_edges[-1], x_edges_values[-1], x_nodes[-1], x_nodes_coord[-1], x_nodes_features[-1], y_edges[-1], edge_cw)
#         y_preds.append(y_pred)


# y_preds = torch.squeeze(torch.stack(y_preds))
# # Plot prediction visualizations
# #plots.plot_predictions(x_nodes_coord, x_edges, x_edges_values, y_edges, y_preds, num_plots=num_samples)



