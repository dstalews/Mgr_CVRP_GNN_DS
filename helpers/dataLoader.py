import numpy as np
import math
from sklearn.utils import shuffle
from scipy.spatial.distance import pdist, squareform, cdist

class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class GoogleCVRPReader(object):
    """Iterator that reads CVRP dataset files and yields mini-batches.
    
    Format expected as in Vinyals et al., 2015: https://arxiv.org/abs/1506.03134, http://goo.gl/NDcOIG
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, shufflecheck = True):
        """
        Args:
            num_nodes: Number of nodes in CVRP tours
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        if shufflecheck:
            self.filedata = shuffle(open(filepath, "r").readlines())  # Always shuffle upon reading data
        else:
            self.filedata = open(filepath, "r").readlines()  
        self.max_iter = (len(self.filedata) // batch_size)

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx])

    def process_batch(self, lines):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_nodes_features = []
        batch_tour_nodes = []
        batch_tour_len = []

        for line_num, line in enumerate(lines):
            line = line.split(" ")  # Split into list
            
            # Compute signal on nodes
            nodes = np.ones(self.num_nodes)  # All 1s for CVRP...
            
            # Convert node coordinates to required format
            nodes_coord = []
            nodes_features = []
            for idx in range(0, 3 * self.num_nodes, 3):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
                nodes_features.append([float(line[idx + 2]),float(line[idx + 2])]) # just capacity
                #nodes_features.append([float(line[idx]), float(line[idx + 1]), float(line[idx + 2])]) # capacity with coordinates
                #x = math.sqrt((nodes_coord[0][0] - float(line[idx]))**2+(nodes_coord[0][1] - float(line[idx + 1]))**2)
                #nodes_features.append([float(line[idx]), float(line[idx + 1]), float(line[idx + 2]), x]) # capacity with coordinates and distance from depot

            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            
            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((self.num_nodes, self.num_nodes))  # Graph is fully connected
            else:
                W = np.zeros((self.num_nodes, self.num_nodes))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections 
                for idx in range(self.num_nodes):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections
            
            # Convert tour nodes to required format
            tour_nodes = []
            tour_nodes_tmp = [int(node) for node in line[line.index('output:') + 1:-1]][:-1]
            for i in range(len(tour_nodes_tmp)):
                if  i + 1 == len(tour_nodes_tmp) or tour_nodes_tmp[i] != tour_nodes_tmp [i+1]:
                    tour_nodes.append(tour_nodes_tmp[i])

            if tour_nodes[-1] != 0:
                tour_nodes.append(tour_nodes_tmp[0])
            
            # Compute node and edge representation of tour + tour_len
            tour_len = 0
            nodes_target = tour_nodes
            edges_target = np.zeros((self.num_nodes, self.num_nodes))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                tour_len += W_val[i][j]
            
            # Add final connection of tour in edge target
            if tour_nodes[0] != j:
                edges_target[j][tour_nodes[0]] = 1
                edges_target[tour_nodes[0]][j] = 1
                tour_len += W_val[j][tour_nodes[0]]
            
            # Concatenate the data
            batch_edges.append(W)
            batch_edges_values.append(W_val)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(nodes_coord)
            batch_nodes_features.append(nodes_features)
            batch_tour_nodes.append(tour_nodes)
            batch_tour_len.append(tour_len)
        
        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)
        batch.edges_values = np.stack(batch_edges_values, axis=0)
        batch.edges_target = np.stack(batch_edges_target, axis=0)
        batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.nodes_features = np.stack(batch_nodes_features, axis=0)
        batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
        batch.tour_len = np.stack(batch_tour_len, axis=0)
        return batch