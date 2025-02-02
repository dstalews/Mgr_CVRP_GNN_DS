import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import numpy as np
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from helpers import GNN, plots, dataLoader, beamSearch, cvrpboard, cvrproute
import time
import pandas as pd
import openpyxl

dtypeFloat = torch.cuda.FloatTensor
dtypeLong = torch.cuda.LongTensor
test_datasets = []
num_neighbors = -1    # when set to -1, it considers all the connections instead of k nearest neighbors
net_names = [f'./model/model_13_350_30_300.pt', f'./model/model_25_400_30_300.pt', f'./model/model_49_300_30_300.pt']
num_nodes_tab = [13,25,49]

for i in range(2,3):

    net = torch.load(net_names[i])
    # Set evaluation mode
    net.eval()

    num_samples = 100
    num_nodes = num_nodes_tab[i]
    num_neighbors = -1
    test_filepath = f'./dataset/CVRP{num_nodes}_val_GNN.txt'
    dataset = iter(dataLoader.GoogleCVRPReader(num_nodes, num_neighbors, 1, test_filepath, False))


    x_edges = []
    x_edges_values = []
    x_nodes = []
    x_nodes_coord = []
    x_nodes_features = []
    y_edges = []
    y_nodes = []
    y_preds = []

    with torch.no_grad():
        for i in range(num_samples):
            sample = next(dataset)
            # Convert batch to torch Variables
            x_edges.append(Variable(torch.LongTensor(sample.edges).type(dtypeLong), requires_grad=False))
            x_edges_values.append(Variable(torch.FloatTensor(sample.edges_values).type(dtypeFloat), requires_grad=False))
            x_nodes.append(Variable(torch.LongTensor(sample.nodes).type(dtypeLong), requires_grad=False))
            x_nodes_coord.append(Variable(torch.FloatTensor(sample.nodes_coord).type(dtypeFloat), requires_grad=False))
            x_nodes_features.append(Variable(torch.FloatTensor(sample.nodes_features).type(dtypeFloat), requires_grad=False))
            y_edges.append(Variable(torch.LongTensor(sample.edges_target).type(dtypeLong), requires_grad=False))
            y_nodes.append(Variable(torch.LongTensor(sample.nodes_target).type(dtypeLong), requires_grad=False))
            
            # Compute class weights
            edge_labels = (y_edges[-1].cpu().numpy().flatten())
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            
            # Forward pass
            y_pred, loss = net.forward(x_edges[-1], x_edges_values[-1], x_nodes[-1], x_nodes_coord[-1], x_nodes_features[-1], y_edges[-1], edge_cw)
            y_preds.append(y_pred)


    y_preds = torch.squeeze(torch.stack(y_preds))
    # Plot prediction visualizations
    #plots.plot_predictions(x_nodes_coord, x_edges, x_edges_values, y_edges, y_preds, num_plots=num_samples)
    y = F.softmax(y_preds, dim=3)  # B x V x V x voc_edges
    y_probs = y[:,:,:,1]  # Prediction probabilities: B x V x V
    for i in range(0,num_samples):
        data_c = tuple(int(x[0]) for x in x_nodes_features[i].tolist()[0])
        test_datasets.append([y_probs[i].tolist(), x_edges_values[i].tolist()[0], data_c, y_nodes[i].tolist()[0], x_nodes_coord[i].tolist()[0], x_edges_values[i].cpu().numpy()])

j=47
td = test_datasets[j]
#print(f'BeamSearch Trasa: {route.croute} Długość: {length}')
print(f'OR Tools Trasa: {td[3]} Długość: {beamSearch.score_length(td[1], td[3])}')
# route2, length2 = cvrpboard.play_game(td[0],td[1],td[2],k)
# print(f'MCTS Trasa: {route2} Długość: {length2}')
# route, length = cvrpboard.play_game(td[0],td[1],td[2],k,'B', beamwidth[j], beamwidth[j]*2)
# print(f'BMCTS Trasa: {route} Długość: {length}')

route = [0, 34, 48, 46, 41, 15, 33, 0, 30, 3, 32, 40, 0, 18, 10, 0, 17, 43, 24, 42, 0, 28, 39, 14, 0, 36, 7, 2, 0, 5, 38, 0, 1, 23, 37, 47, 8, 0, 45, 26, 6, 0, 12, 29, 0, 16, 22, 0, 27, 44, 0, 11, 4, 0, 19, 31, 0, 25, 9, 0, 21, 20, 13, 0, 35, 0]




plots.plot_two_routes(td[4], td[5], td[3], route, file_name=f'./plots_and_data/CVRP_{len(set(route))}_{j}_testy_MCTS_tests.png')

