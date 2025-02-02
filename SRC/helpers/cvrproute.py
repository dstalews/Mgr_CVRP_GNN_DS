class CVRPRoute:
    def __init__(self, num_nodes, data_p):
        self.num_nodes = num_nodes
        self.target = (False,) * num_nodes * num_nodes
        self.croute = [0]
        self.cap = 0
        self.data_p = data_p
        self.score = 0.0
        self.prior_p = 0

    def __hash__(self):
       return hash(self.target)

    def __eq__(self, other):
        return self.target == other.target

    def add_node(self, node, c): 
        self.score = self.score - self.data_p[node][self.croute[-1]]
        if self.data_p[node][self.croute[-1]] > 0:
            self.prior_p = self.data_p[node][self.croute[-1]]
        if self.data_p[node][0] > 0.0000002:    
            self.data_p[node][0] = 0.0000002
            self.data_p[0][node] = 0.0000002
        self.data_p[node][self.croute[-1]] = 0

        new_index = int(self.croute[-1]*self.num_nodes + node)
        self.target = self.target[:new_index] + (True,) + self.target[new_index + 1 :]
        self.croute.append(node)        
        
        if node != 0:
            self.cap = self.cap + c
        else:
            self.cap = 0


    def check_next_node(self, node, c):
        if (self.cap + c <= 15 and node not in self.croute) or (node == 0 and self.croute[-1] != 0):
            return True
        else:
            return False
        
    def get_next_node(self, node, uq_nodes):
        index = node * self.num_nodes
        for i in range(index, index + self.num_nodes):
            if self.target[i] and (i - index == 0 or i - index not in uq_nodes):
                return i - index
        
        return None
    
    def is_terminal_route(self):
        if len(set(self.croute)) == self.num_nodes and self.croute[-1]==0:
            return True
        return False