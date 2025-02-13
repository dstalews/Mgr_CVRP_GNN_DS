import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import time

from sklearn.utils.class_weight import compute_class_weight

from torch_geometric.nn import TransformerConv
from . import dataLoader

dtypeFloat = torch.float
dtypeLong = torch.long

class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)

        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
        return e_bn
    
#@title MLP layer
class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y
    
class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H
        Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        gateVx = edge_gate * Vx  # B x V x V x H
        if self.aggregation=="mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
        
    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  # Extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = Ue + Vx + Wx
        return e_new
    
class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H
        # Compute edge gates
        edge_gate = torch.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate)
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new
    
class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGatedGCNModel, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config['num_nodes']
        self.node_dim = config['node_dim']
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['voc_nodes_out'] 
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        # Node and edge embedding layers/lookups
       
        # We are using TransformerConv layer from torch geometric library!
        self.nodes_coord_embedding = TransformerConv(self.node_dim, self.hidden_dim) 
      
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def loss_edges(self, y_pred_edges, y_edges, edge_cw):
        """
        Loss function for edge predictions.

        Args:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss

        Returns:
            loss_edges: Value of loss function
        
        """
        # Edge loss
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        loss_edges = nn.NLLLoss(edge_cw)
        loss = loss_edges(y.contiguous(), y_edges)
        return loss

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, x_nodes_features, y_edges, edge_cw):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
            x_nodes_features: Input node coordinates with capacities (batch_size, num_nodes, node_dim)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss
            # y_nodes: Targets for nodes (batch_size, num_nodes, num_nodes)
            # node_cw: Class weights for nodes loss

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            # y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
            loss: Value of loss function
        """
        # Node and edge embedding
        edge_index = torch.squeeze(x_edges).nonzero().t().contiguous()
        x = self.nodes_coord_embedding(torch.squeeze(x_nodes_features), edge_index)  
        x = torch.unsqueeze(x, 0)
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # B x V x H, B x V x V x H
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out
        
        # Compute loss
        edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)  # Convert to tensors
        loss = self.loss_edges(y_pred_edges.cuda(), y_edges.cuda(), edge_cw)
        
        return y_pred_edges, loss
    
def train_one_epoch(net, optimizer, config):
    # Set training mode
    net.train()

    # Assign parameters
    num_nodes = config['num_nodes']
    num_neighbors = config['num_neighbors']
    batches_per_epoch = config['batches_per_epoch']
    accumulation_steps = config['accumulation_steps']
    train_filepath = config['train_filepath']

    # Load cvrp data
    dataset = dataLoader.GoogleCVRPReader(num_nodes, num_neighbors, 1, train_filepath)
    if batches_per_epoch != -1:
        batches_per_epoch = min(batches_per_epoch, dataset.max_iter)
    else:
        batches_per_epoch = dataset.max_iter

    # Convert dataset to iterable
    dataset = iter(dataset)
    
    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    running_nb_data = 0

    start_epoch = time.time()
    for batch_num in range(batches_per_epoch):
        # Generate a batch of cvrps
        try:
            batch = next(dataset)
        except StopIteration:
            break

        # Convert batch to torch Variables
        x_edges = Variable(torch.tensor(data=batch.edges,dtype=dtypeLong,device='cuda'), requires_grad=False)
        x_edges_values = Variable(torch.tensor(data=batch.edges_values,dtype=dtypeFloat,device='cuda'), requires_grad=False)
        x_nodes = Variable(torch.tensor(data=batch.nodes,dtype=dtypeLong,device='cuda'), requires_grad=False)
        x_nodes_coord = Variable(torch.tensor(data=batch.nodes_coord,dtype=dtypeFloat,device='cuda'), requires_grad=False)
        x_nodes_features = Variable(torch.tensor(data=batch.nodes_features,dtype=dtypeFloat,device='cuda'), requires_grad=False)
        y_edges = Variable(torch.tensor(data=batch.edges_target,dtype=dtypeLong,device='cuda'), requires_grad=False)
        y_nodes = Variable(torch.tensor(data=batch.nodes_target,dtype=dtypeLong,device='cuda'), requires_grad=False)
        
        # Compute class weights (if uncomputed)
        if type(edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        
        # Forward pass
        y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, x_nodes_features, y_edges, edge_cw)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        loss.backward()

        # Backward pass
        if (batch_num+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update running data
        running_nb_data += 1
        running_loss += loss.data.item()* accumulation_steps  # Re-scale loss
        
    # Compute statistics for full epoch
    loss = running_loss/ running_nb_data

    return time.time()-start_epoch, loss

def test(net, config, mode='test'):
    # Set evaluation mode
    net.eval()

    # Assign parameters
    num_nodes = config['num_nodes']
    num_neighbors = config['num_neighbors']
    batches_per_epoch = 1 #config['batches_per_epoch']
    val_filepath = config['val_filepath']
    test_filepath = config['test_filepath']

    # Load cvrp data
    if mode == 'val':
        dataset = dataLoader.GoogleCVRPReader(num_nodes, num_neighbors, 1, filepath=val_filepath)
    elif mode == 'test':
        dataset = dataLoader.GoogleCVRPReader(num_nodes, num_neighbors, 1, filepath=test_filepath)

    # Convert dataset to iterable
    dataset = iter(dataset)
    
    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    running_nb_data = 0
    
    with torch.no_grad():
        start_test = time.time()
        for batch_num in range(batches_per_epoch):
            # Generate a batch of cvrps
            try:
                batch = next(dataset)
            except StopIteration:
                break

            # Convert batch to torch Variables
            x_edges = Variable(torch.tensor(data=batch.edges,dtype=dtypeLong,device='cuda'), requires_grad=False)
            x_edges_values = Variable(torch.tensor(data=batch.edges_values,dtype=dtypeFloat,device='cuda'), requires_grad=False)
            x_nodes = Variable(torch.tensor(data=batch.nodes,dtype=dtypeLong,device='cuda'), requires_grad=False)
            x_nodes_coord = Variable(torch.tensor(data=batch.nodes_coord,dtype=dtypeFloat,device='cuda'), requires_grad=False)
            x_nodes_features = Variable(torch.tensor(data=batch.nodes_features,dtype=dtypeFloat,device='cuda'), requires_grad=False)
            y_edges = Variable(torch.tensor(data=batch.edges_target,dtype=dtypeLong,device='cuda'), requires_grad=False)
            y_nodes = Variable(torch.tensor(data=batch.nodes_target,dtype=dtypeLong,device='cuda'), requires_grad=False)
            
            # Compute class weights (if uncomputed)
            if type(edge_cw) != torch.Tensor:
                edge_labels = y_edges.cpu().numpy().flatten()
                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

            # Forward pass
            y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, x_nodes_features, y_edges, edge_cw)
            loss = loss.mean()  # Take mean of loss across multiple GPUs

            # Update running data
            running_nb_data += 1
            running_loss += loss.data.item()
            
    # Compute statistics for full epoch
    loss = running_loss/ running_nb_data

    return time.time()-start_test, loss

def update_learning_rate(optimizer, lr):
  """
  Updates learning rate for given optimizer.

  Args:
      optimizer: Optimizer object
      lr: New learning rate

  Returns:
      optimizer: Updated optimizer objects
  """
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr
  return optimizer



