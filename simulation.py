from collections import defaultdict
import scipy as sp
import numpy as np
import networkx as nx
import random
import copy
import json
import os
import matplotlib.pyplot as plt

"""
Node class stores node attributes: 
    - idx, which corresponds to its node index in graph.
    - color, which corresponds to its group index.
    - embed, the node's embedding vector.
    - age, the node's age.
    - time, the duration of time it has been added to the graph.
"""
class Node:
    def __init__(self, idx, color=None, embed=None, age=None, is_treatment=False):
        self.idx = idx
        self.color = color
        self.embed = embed
        self.age = age
        self.time = 0
        self.is_treatment = is_treatment
    
    def generate_embedding(self, mean, variance):
        self.embed = np.random.multivariate_normal(mean, variance)

"""
Group class is a collection of Nodes with attributes:
    - mean and variance, for generating node embeddings.
    - size, its current number of nodes.
    - color, its index among the group list.
    - node_map, a dictionary that maps node index to its Node object.
"""
class Group:
    def __init__(self, mean: np.ndarray, variance: np.ndarray, color: int, node_ids: 'list[int]'):
        assert len(set(node_ids)) == len(node_ids)
        self.mean = mean
        self.variance = variance
        self.size = len(node_ids)
        self.color = color
        self.node_map = defaultdict(Node)

        self.initialize_nodes(node_ids)

    def initialize_nodes(self, node_ids: 'list[int]'):
        embeddings = np.random.multivariate_normal(self.mean, self.variance, self.size)
        for i in range(self.size):
            node = Node(node_ids[i], color=self.color, embed=embeddings[i], age=self.sample_age_distribution())
            self.node_map[node_ids[i]] = node
    
    def add_node(self, idx):
        # do we add is_treatment property here?
        assert idx not in self.node_map
        embedding = np.random.multivariate_normal(self.mean, self.variance)
        node = Node(idx, color=self.color, embed=embedding, age=self.sample_age_distribution())
        self.node_map[idx] = node
        self.size += 1
        return node
    
    def remove_node(self, idx):
        assert idx in self.node_map
        self.node_map.pop(idx)
        self.size -= 1
    
    def sample_age_distribution(self, min_age=18, max_age=65):
        return np.random.randint(min_age, max_age)  # adult to retirement
    
    def time_increment(self):
        for node_idx, node_obj in self.node_map.items():
            node_obj.time += 1
            node_obj.age += 1

"""
Network class takes in a list of Group objects to create a graph based on their nodes information. It supports graph-level operations, stores graph attributes, and provides a snapshot of graph metrics.
    - edge_matrix: a matrix that stores the number of connections for group pairs. edge_matrix[i][j] refers to the number of edges between group i and group j. 
    - edge_phase: a dictionary that maps 5 different steps (initialization, natural growth phase 1, natural growth phase 2 mediated & unmediated, recommendation) to the number of edges that are formed in the corresponding steps.
    - edge_matrix_phase: a dictionary that maps 5 different steps to the connection matrixes that keep record of the number of connections for group pairs.

"""
class Network:
    def __init__(self, group_lst: 'list[Group]', **kwargs):
        self.a = kwargs.get('a', 2)
        self.b = kwargs.get('b', 5)
        self.conn_matrix = kwargs.get('conn_matrix', None)
        self.time_step = 0

        self.num_groups = len(group_lst)
        self.group_lst = group_lst
        self.init_sizes = [grp.size for grp in group_lst]
        self.G = nx.Graph()

        self.edge_matrix = np.zeros((self.num_groups, self.num_groups))
        self.edge_matrix_0 = np.zeros((self.num_groups, self.num_groups))
        self.edge_matrix_1 = np.zeros((self.num_groups, self.num_groups))
        self.edge_matrix_2_mediated = np.zeros((self.num_groups, self.num_groups))
        self.edge_matrix_2_unmediated = np.zeros((self.num_groups, self.num_groups))
        self.edge_matrix_3 = np.zeros((self.num_groups, self.num_groups))
        self.edge_matrix_phase = {'init': self.edge_matrix_0, 'ngp1': self.edge_matrix_1, 'ngp2_unmediated': self.edge_matrix_2_unmediated, 'ngp2_mediated': self.edge_matrix_2_mediated, 'rec': self.edge_matrix_3}
        self.edge_phase = {'init': 0, 'ngp1': 0, 'ngp2_unmediated': 0, 'ngp2_mediated': 0, 'rec': 0}
        self.initialize_node()

        self.init_how = kwargs.get('init_how', 'color')
        self.initialize_edge(self.init_how)

    # initilize the nodes in graph by group_lst
    def initialize_node(self):
        for grp in self.group_lst:
            for node_idx, node_obj in grp.node_map.items():
                self.G.add_node(node_obj.idx, vector=node_obj.embed, color=node_obj.color, age=node_obj.age, time=node_obj.time)

    # initialize the edges in graph by constant connection probability (color) or embedding affinity (embedding)
    def initialize_edge(self, how):
        if how == 'color':
            self._initialize_edge_by_color()
        if how == 'embedding':
            self._initialize_edge_by_embedding()
    
    def _initialize_edge_by_color(self):
        assert self.conn_matrix is not None
        for i in range(self.num_groups):
            g1 = self.group_lst[i]
            for j in range(i, self.num_groups):
                g2 = self.group_lst[j]
                prob = self.conn_matrix[g1.color][g2.color]
                # intra-group: n*(n-1)/2 types of combination
                if g1.color == g2.color:
                    gs = g1.size
                    sample = np.random.binomial(1, prob, (gs, gs))
                    conn_pairs = np.where(sample == 1)
                    node_lst = list(g1.node_map.keys())
                    # only cares (n1, n2) not (n2, n1)
                    conn_pairs = [[node_lst[pair[0]], node_lst[pair[1]]] for pair in conn_pairs if pair[0] < pair[1]]
                    [self.add_edge(pair[0], pair[1], 'init') for pair in conn_pairs]
                    
                # inter-group: n1*n2 types of combination
                else:
                    sample = np.random.binomial(1, prob, (g1.size, g2.size))
                    conn_pairs = np.where(sample == 1)
                    g1_node_lst = list(g1.node_map.keys())
                    g2_node_lst = list(g2.node_map.keys())
                    conn_pairs = [[g1_node_lst[pair[0]], g2_node_lst[pair[1]]] for pair in conn_pairs]
                    [self.add_edge(pair[0], pair[1], 'init') for pair in conn_pairs]
                    
    def _initialize_edge_by_embedding(self):
        nodes = self.G.nodes
        # n by d matrix of embeddings
        embedding_mat = np.array([node_info['vector'] for _, node_info in nodes.data()])
        inner_prod_matrix = embedding_mat @ embedding_mat.T
        pairwise_prob = np.triu(self.sigmoid(inner_prod_matrix), 1)
        sample = np.random.binomial(1, pairwise_prob, (self.G.number_of_nodes(), self.G.number_of_nodes()))
        sparse_sample = sp.sparse.csr_matrix(sample)
        l1, l2 = sparse_sample.nonzero()
        conn_pairs = list(zip(l1, l2))
        [self.add_edge(pair[0], pair[1], 'init') for pair in conn_pairs]
    
        
    def add_node(self, node: Node):
        assert not self.G.has_node(node.idx)
        if node.embed is None:
            grp = self.group_lst[node.color]
            node.embed = np.random.multivariate_normal(grp.mean, grp.variance)
        self.G.add_node(node.idx, vector=node.embed, color=node.color, age=node.age, time=node.time, obj=node)
        
    def add_edge(self, n1: int, n2: int, phase: str):
        assert not self.G.has_edge(n1, n2)
        self.G.add_edge(n1, n2, phase=phase, time=self.time_step)
        self.edge_phase[phase] += 1
        c1, c2 = self.G.nodes[n1]['color'], self.G.nodes[n2]['color']
        if c1 == c2:
            self.edge_matrix[c1][c2] += 1
            self.edge_matrix_phase[phase][c1][c2] += 1
        else:
            self.edge_matrix[c1][c2] += 1
            self.edge_matrix[c2][c1] += 1
            self.edge_matrix_phase[phase][c1][c2] += 1
            self.edge_matrix_phase[phase][c2][c1] += 1
    
    def remove_edge(self, n1, n2):
        c1, c2 = self.G.nodes[n1]['color'], self.G.nodes[n2]['color']
        edge_phase = self.G[n1][n2]['phase']
        self.G.remove_edge(n1, n2)
        self.edge_phase[edge_phase] -= 1
        if c1 == c2:
            self.edge_matrix[c1][c2] -= 1
            self.edge_matrix_phase[edge_phase][c1][c2] -= 1
        else:
            self.edge_matrix[c1][c2] -= 1
            self.edge_matrix[c2][c1] -= 1
            self.edge_matrix_phase[edge_phase][c1][c2] -= 1
            self.edge_matrix_phase[edge_phase][c2][c1] -= 1

    def remove_node(self, idx):
        for n1, n2, data in list(self.G.edges(idx, data=True)):
            self.remove_edge(n1, n2)
        self.G.remove_node(idx)
    
    def time_increment(self):
        for node in list(self.G.nodes):
            self.G.nodes[node]['age'] += 1
            self.G.nodes[node]['time'] += 1
        self.time_step += 1
            
    def get_adamic_adar_list(self, pairs):
        adamic_adar_triples = nx.adamic_adar_index(self.G, pairs)
        return [(u, v, p) for u, v, p in adamic_adar_triples]

    def get_dist_nodes(self, node, dist):
        return nx.descendants_at_distance(self.G, node, dist)
    
    def get_treatment_nodes(self):
        return [n for n in self.G.nodes if self.G.nodes[n]['is_treatment']]
    
    def get_control_nodes(self):
        return [n for n in self.G.nodes if not self.G.nodes[n]['is_treatment']]
    
    def assign_treatment(self, idx_list):
        # for idx in idx_list:
        #     self.G.nodes[idx]['is_treatment'] = True
        raise NotImplementedError
    
    def remove_treatment(self, idx_list):
        # for idx in idx_list:
        #     self.G.nodes[idx]['is_treatment'] = False
        raise NotImplementedError
    
    @property
    def edges(self):
        return self.G.number_of_edges()

    @property
    def nodes(self):
        return self.G.number_of_nodes()
    
    @property
    def treatment_nodes(self):
        # how many treatment nodes are there
        raise NotImplementedError

    def global_clustering(self, grp=None, idx_list=None):
        # handle idx_list for every metric
        if grp:
            color = grp.color
            clustering_coeff = [nx.algorithms.cluster.clustering(self.G, n) for n in self.G.nodes() if self.G.nodes[n]['color'] == color]
            return sum(clustering_coeff) / grp.size
        
        clustering_coeff = [nx.algorithms.cluster.clustering(self.G, n) for n in self.G.nodes()]
        return sum(clustering_coeff) / self.nodes
    
    def betweenness_centrality(self, rescale=True):
        centralities = nx.betweenness_centrality(self.G, normalized=True)
        if rescale:
            vals = list(centralities.values())
            min_val = min(vals)
            max_val = max(vals)
            centralities = {k: (v - min_val) / (max_val - min_val) for k, v in centralities.items()}
        return centralities

    def avg_degree(self, grp=None):
        degree_list = list(self.G.degree())
        if grp:
            color = grp.color
            degree_list = [(n,d) for n, d in degree_list if self.G.nodes[n]['color'] == color]
            
        return np.mean([i[1] for i in degree_list])

    def degree_var(self, grp=None):
        degree_list = list(self.G.degree())
        if grp:
            color = grp.color
            degree_list = [(n,d) for n, d in degree_list if self.G.nodes[n]['color'] == color]
        return np.mean([(i[1] - self.avg_degree(grp)) ** 2 for i in degree_list])
    
    @property
    def bi_frac(self):
        return np.triu(self.edge_matrix, 1).sum() / self.edges
    
    @property
    def avg_age(self):
        age_list = nx.get_node_attributes(self.G, "age").values()
        return sum(age_list) / len(age_list)
    
    @property
    def avg_nodes_dist2(self):
        nn = [len(nx.descendants_at_distance(self.G, i, 2))
              for i in self.G.nodes()]
        return sum(nn)/len(nn)
    
    def mono_bi_edge_phase(self, phase):
        edge_matrix = self.edge_matrix_phase[phase]
        return edge_matrix[0][0] + edge_matrix[1][1], edge_matrix[0][1]

    def gini_coeff(self, nodes_subset=None, grp=None):
        if nodes_subset:
            raise NotImplementedError
        degree_list = list(self.G.degree())
        if grp:
            color = grp.color
            degree_list = [(n,d) for n, d in degree_list if self.G.nodes[n]['color'] == color]
        degree_list = [d for _, d in degree_list]
        degree_list = np.sort(degree_list)
        n = len(degree_list)
        gini = 2 * np.sum( np.arange(n) * degree_list) / (n * np.sum(degree_list)) - (n + 1) / n
        return gini
        
    def homophily(self, grp: Group):
        color = grp.color
        return self.edge_matrix[color][color] / np.sum(self.edge_matrix, axis=0)[color] - grp.size / self.nodes
    
    def sigmoid(self, n):
        return 1/(1 + np.exp(-n * self.a + self.b))

    def plot_graph(self, size_by = 'degree'):
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.clf()
        # plot scatter of points
        colors = dict(self.G.nodes(data="color"))
        degrees = dict(self.G.degree())
        centralities = dict(self.betweenness_centrality().items())
        mean_degree = np.mean(list(degrees.values()))
        for node_id, c in colors.items():
            coord = self.G.nodes[node_id]["vector"]
            if size_by == 'degree':
                size = degrees[node_id]*5/mean_degree
            elif size_by == 'betweenness':
                size = (0.1+centralities[node_id])*10
            else:
                size = 10
            if c == 0:
                plt.plot(coord[0], coord[1], marker ='o', markersize = size, color = "red")
            elif c == 1:
                plt.plot(coord[0], coord[1], marker ='s', markersize = size, color = "blue")
        # plot edges
        edges = list(self.G.edges)
        for edge in edges:
            node1, node2 = edge
            node1_coord = self.G.nodes[node1]["vector"]
            node2_coord = self.G.nodes[node2]["vector"]

            if (self.G.nodes[node1]["color"] != self.G.nodes[node2]["color"]):
                color = 'g'
            else:
                color = 'k'
            plt.plot([node1_coord[0], node2_coord[0]], [node1_coord[1], node2_coord[1]],
                    '-', color=color, lw=0.2)
        plt.axis('equal')
        plt.show()
    
"""
Dynamics class takes in a Network object and experiment-specific parameters to run one step of the experiment.
"""
class Dynamics:
    def __init__(self, network: Network, **kwargs):
        self.network = network
        self.grp_lst = network.group_lst
        self.a = kwargs.get('a', 2)
        self.b = kwargs.get('b', 5)
        self.death_func = kwargs.get('death_func', lambda x: 0.0001*np.exp(0.08*x))
        self.num_rec = 0
        self.num_acc = 0
        self.ng_how = kwargs.get('ng_how', 'color')
        self.p2_prob = kwargs.get('p2_prob', 0.05)
        self.p2_mediated = kwargs.get('p2_mediated', True)
        self.Ns = kwargs.get('Ns', None)
        self.Nf = kwargs.get('Nf', None)
        self.node_removal = kwargs.get('node_removal', True)

    def step(self, nodes_step, idx, intervention, **kwargs):
        # assume new node comes in proportional to groups' initial sizes
        arrival_probs = self.network.init_sizes/np.sum(self.network.init_sizes)
        new_nodes_grp = np.random.choice(len(self.grp_lst), size=nodes_step, p=arrival_probs)
        for i in range(nodes_step):
            grp = self.grp_lst[new_nodes_grp[i]]
            node = grp.add_node(idx)
            self.step_natural_growth(node, self.ng_how, self.p2_mediated)
            idx += 1
        if intervention:
            self.intervention(**kwargs)
        self.time_increment()
        if self.node_removal:
            self.remove_node()
        return idx
    
    def triadic_closure_init(self):
        node_lst = self.network.G.nodes
        for idx in node_lst:
            dist2_nodes = list(self.network.get_dist_nodes(idx, 2))
            random.shuffle(dist2_nodes)
            if not (self.Nf is None):
                dist2_nodes = dist2_nodes[:self.Nf]
            for i in dist2_nodes:
                accept = self.accept_edge(idx, i, how='constant', acc_prob=self.p2_prob)
                if accept:
                    self.network.add_edge(idx, i, 'init')

    def step_natural_growth(self, node: Node, ng_how, p2_mediated):
        self.network.add_node(node)
        # phase 1
        node_list = copy.deepcopy(list(self.network.G.nodes))
        random.shuffle(node_list)
        if not (self.Ns is None):
            node_list = node_list[:self.Ns]
        for i in node_list:
            if ng_how == 'color':
                p1_prob = self.network.conn_matrix
            else:
                p1_prob = None
            accept = self.accept_edge(node.idx, i, how=ng_how, acc_prob=p1_prob)
            if accept:
                self.network.add_edge(node.idx, i, 'ngp1')
        # phase 2
        dist2_nodes = list(self.network.get_dist_nodes(node.idx, 2))
        random.shuffle(dist2_nodes)
        if not p2_mediated:
            dist2_nodes = self.remove_mediating_nodes(node.idx, dist2_nodes)
        if not (self.Nf is None):
            dist2_nodes = dist2_nodes[:self.Nf]

        if not p2_mediated:
            for i in dist2_nodes:
                accept = self.accept_edge(node.idx, i, how='constant', acc_prob=self.p2_prob)
                if accept:
                    self.network.add_edge(node.idx, i, 'ngp2_unmediated')
        else:
            edge_phases = nx.get_edge_attributes(self.network.G, 'phase')
            for i in dist2_nodes:
                accept = self.accept_edge(node.idx, i, how='constant', acc_prob=self.p2_prob)
                if accept:
                    mediated = True
                    for common_neighbor in nx.common_neighbors(self.network.G, node.idx, i):
                        if (common_neighbor, i) in edge_phases:
                            if edge_phases[(common_neighbor, i)] != 'rec':
                                mediated = False
                                break
                        else:
                            if edge_phases[(i, common_neighbor)] != 'rec':
                                mediated = False
                                break
                    if mediated:
                        self.network.add_edge(node.idx, i, 'ngp2_mediated')
                    else:
                        self.network.add_edge(node.idx, i, 'ngp2_unmediated')
    
    def intervention(self, **kwargs): 
        # intervention is applied only to treatment nodes  
        self.rec_how = kwargs.get('rec_how', 'random_fof')
        self.rec_sample_size = kwargs.get('rec_sample_size', None)
        self.rec_sample_fraction = kwargs.get('rec_sample_fraction', None)
        self.acc_how = kwargs.get('acc_how', 'embedding')
        self.rec_acc_prob = kwargs.get('acc_prob', None)
        self.rec_distance = kwargs.get('rec_distance', 2)
        self.edge_removal = kwargs.get('edge_removal', False)

        # sample nodes
        if self.rec_sample_size != None:
            node_lst = copy.deepcopy(list(self.network.G.nodes))
            random.shuffle(node_lst)
            node_lst = node_lst[:self.rec_sample_size]
        elif self.rec_sample_fraction != None:
            node_lst = copy.deepcopy(list(self.network.G.nodes))
            random.shuffle(node_lst)
            rec_sample_size = int(self.rec_sample_fraction * self.network.nodes)
            node_lst = node_lst[:rec_sample_size]
        else: 
            node_lst = self.network.G.nodes
        # filter nodes that are not treatment nodes
        # node_lst = [node for node in node_lst if self.network.G.nodes[node]['is_treatment'] == True]
        for idx in node_lst:
            candidate = self.recommend_edge(idx, self.rec_how, **kwargs)
            if candidate != None:
                self.num_rec += 1
                accept_rec = self.accept_edge(idx, candidate, self.acc_how, self.rec_acc_prob)
                if accept_rec:
                    self.num_acc += 1
                    if self.edge_removal:
                        # remove an edge
                        # for now remove a random edge
                        self._remove_edge_random(idx)
                    self.network.add_edge(idx, candidate, 'rec')
                    
                
    def accept_edge(self, idx, candidate, how, acc_prob):
        """Return True if the edge is accepted, False otherwise"""
        if how == 'constant':
            return(self._accept_edge_constant(idx, candidate, acc_prob))
        if how == 'color':
            return(self._accept_edge_group_based(idx, candidate, acc_prob))
        if how == 'embedding':
            return(self._accept_edge_embedding_based(idx, candidate, acc_prob))
    
    def _accept_edge_constant(self, idx, candidate, acc_prob):
        assert acc_prob is not None
        return(np.random.random_sample() < acc_prob)
    
    def _accept_edge_group_based(self, idx, candidate, acc_prob):
        assert acc_prob is not None
        color_idx = self.network.G.nodes[idx]['color']
        color_candidate = self.network.G.nodes[candidate]['color']
        return(np.random.random_sample() < acc_prob[color_idx][color_candidate])
    
    def _accept_edge_embedding_based(self, idx, candidate, acc_prob):
        if acc_prob is None:
            acc_prob = 1
        # here acc_prob acts as a scaling factor
        vector_candidate = self.network.G.nodes[candidate]['vector']
        vector_idx = self.network.G.nodes[idx]['vector']
        prob = acc_prob*self.sigmoid(np.inner(vector_candidate, vector_idx))
        return(np.random.random_sample() < prob)
    
    def _remove_edge_random(self, idx):
        # remove a random edge
        nns = list(self.network.G.neighbors(idx))
        if len(nns) > 0:
            random.shuffle(nns)
            self.network.remove_edge(idx, nns[0])

    def recommend_edge(self, idx, how = 'random_fof', **kwargs):
        if how == 'random_fof':
            return(self._recommend_random_fof(idx))
        if how == 'adamic_adar':
            return(self._recommend_adamic_adar(idx))
        if how == 'embedding':
            return(self._recommend_embedding(idx, **kwargs))
    
    def _recommend_random_fof(self, idx):
        assert self.rec_distance >= 2
        nns = set()
        for d in range(2, self.rec_distance+1):
            nns = nns.union(self.network.get_dist_nodes(idx, d))
        if (len(nns) == 0):
            return None
        return(random.sample(list(nns), 1)[0])
    
    def _recommend_adamic_adar(self, idx):
        assert self.rec_distance >= 2
        nns = set()
        for d in range(2, self.rec_distance+1):
            nns = nns.union(self.network.get_dist_nodes(idx, d))
        if (len(nns) == 0):
            return None
        pairs = [(idx, other) for other in nns]
        aai = self.network.get_adamic_adar_list(pairs)
        aai = sorted(aai, key=lambda triple: triple[2], reverse=True)
        candidate = aai[0][1] if idx == aai[0][0] else aai[0][0]
        return candidate
    
    def _recommend_embedding(self, idx, beta=10, **kwargs):
        neighbors = set(self.network.G.neighbors(idx))
        eligible = list(set(self.network.G.nodes).difference(neighbors))
        idx_embedding = self.network.G.nodes[idx]['vector']
        embedding_mat = np.array([self.network.G.nodes[i]['vector'] for i in eligible])
        similarity = embedding_mat @ idx_embedding
        softmax_probs = sp.special.softmax(similarity*beta)
        threshold = 0.1 / self.network.nodes
        softmax_probs = softmax_probs*(softmax_probs > threshold)
        if np.sum(softmax_probs) == 0:
            return None
        softmax_probs = softmax_probs/np.sum(softmax_probs)
        candidate = np.random.choice(eligible, p=softmax_probs)
        return candidate
        
    def time_increment(self):    
        for grp in self.grp_lst:
            grp.time_increment()
        self.network.time_increment()

    def remove_node(self):
        ages = list(self.network.G.nodes(data='age'))
        idxs = np.array([x[0] for x in ages])
        death_prob = np.array([self.death_func(x[1]) for x in ages])
        sample = np.random.random_sample(len(ages))
        deaths = idxs[sample < death_prob]
        for idx in deaths:
            grp = self.grp_lst[self.network.G.nodes[idx]['color']]
            self.network.remove_node(idx)
            grp.remove_node(idx)
    
    def remove_mediating_nodes(self, idx, nns):
        edge_phases = nx.get_edge_attributes(self.network.G, 'phase')
        removed_nns = nns.copy()
        for nn in nns:
            indirect = True
            for common_neighbor in nx.common_neighbors(self.network.G, idx, nn):
                if (common_neighbor, nn) in edge_phases:
                    if edge_phases[(common_neighbor, nn)] != 'rec':
                        indirect = False
                        break
                else:
                    if edge_phases[(nn, common_neighbor)] != 'rec':
                        indirect = False
                        break
            if indirect:
                removed_nns.remove(nn)
        return removed_nns
    
    def sigmoid(self, n):
        return 1/(1 + np.exp(-n * self.a + self.b))

"""
Experiment class is where we initialize the Groups and their Nodes, the Network, and the Dynamics. 

It runs the Dynamics through the entire time steps, and records the graph metrics along the way.
"""
class Experiment:
    def __init__(self, grp_info: 'list[dict]', **kwargs):
        self.latest_idx = 0
        self.seed = kwargs.get('seed', 0)
        self.node_step = kwargs.get('node_step', 20)
        self.total_step = kwargs.get('total_step', 200)
        
        np.random.seed(self.seed)
        random.seed(self.seed)

        group_lst = []
        for i in range(len(grp_info)):
            grp_dict = grp_info[i]
            grp = Group(mean=grp_dict['mean'], variance=grp_dict['variance'], color=i, node_ids=range(self.latest_idx, self.latest_idx+grp_dict['size']))
            group_lst.append(grp)
            self.latest_idx += grp_dict['size']
        
        self.dynamics = Dynamics(Network(group_lst, **kwargs), **kwargs)

    def run(self, **kwargs):
        self.intervention_time = kwargs.get('intervention_time', list(range(self.total_step)))
        self.compute_metrics(0, initialize=True, **kwargs)
        self.dynamics.triadic_closure_init()
        for t in range(1, self.total_step+1):
            if t in self.intervention_time:
                step_intervention = True
                # make all nodes treament nodes
                # all_nodes = list(self.dynamics.network.G.nodes)
                # self.dynamics.network.assign_treatment(all_nodes)
            else:
                step_intervention = False
            self.latest_idx = self.dynamics.step(self.node_step, self.latest_idx, step_intervention, **kwargs)
            self.compute_metrics(t, **kwargs)

    def compute_metrics(self, time_step, **kwargs):
        freq = kwargs.get('freq', 1)

        compute_nodes = kwargs.get('comp_nodes', True)
        compute_edges = kwargs.get('comp_edges', True)
        compute_degree = kwargs.get('comp_degree', True)
        compute_degree_var = kwargs.get('comp_degvar', True)
        compute_bi_frac = kwargs.get('comp_bifrac', True)
        compute_cluster = kwargs.get('comp_cluster', True)
        compute_age = kwargs.get('comp_age', True)
        compute_nn = kwargs.get('comp_nn', True)
        compute_homophily = kwargs.get('comp_homop', True)
        compute_gini = kwargs.get('comp_gini', True)
        compute_num_rec = kwargs.get('comp_num_rec', True)
        compute_num_acc = kwargs.get('comp_num_acc', True)
        compute_grp_metrics = kwargs.get('comp_grp_metrics', False)
        
        #metric_names = ['nodes', 'edges', 'avg_degree', 'degree_variance', 'bi_frac', 'global_clustering', 'avg_age', 'avg_nn', 'homophily', 'gini', 'num_rec', 'num_acc']
        # for metric in metric_names:
        #     bla

        if time_step == 0:
            
            self.metrics = defaultdict(list)
            self.metrics['time'] = []
            self.metrics['added_nodes'] = []
            self.metrics['nodes'] = []
            self.metrics['edges'] = []
            phases = ['phase0', 'phase1', 'phase2_unmediated', 'phase2_mediated', 'phase3']
            for p in phases:
                self.metrics[p] = []
                self.metrics[f"{p}_mono"] = []
                self.metrics[f"{p}_bi"] = []
            self.metrics['avg_degree'] = []
            self.metrics['degree_variance'] = []
            self.metrics['bi_frac'] = []
            self.metrics['global_clustering'] = []
            self.metrics['avg_age'] = []
            self.metrics['avg_nn'] = []
            self.metrics['gini_coeff'] = []
            self.metrics['num_rec'] = []
            self.metrics['num_acc'] = []
            for i in range(len(self.dynamics.grp_lst)):
                self.metrics[f"nodes_{i}"] = []
                self.metrics[f"homophily_{i}"] = []
                self.metrics[f"avg_degree_{i}"] = []
                self.metrics[f"degree_variance_{i}"] = []
                self.metrics[f"global_clustering_{i}"] = []
                self.metrics[f"gini_coeff_{i}"] = []
        
        if time_step == 1:
            self.metrics['a'] = self.dynamics.a
            self.metrics['b'] = self.dynamics.b
            self.metrics['init_how'] = self.dynamics.network.init_how
            self.metrics['conn_matrix'] = self.dynamics.network.conn_matrix
            if self.metrics['conn_matrix'] is not None:
                self.metrics['conn_matrix'] = self.metrics['conn_matrix'].tolist()
            self.metrics['ng_how'] = self.dynamics.ng_how
            self.metrics['p2_prob'] = self.dynamics.p2_prob
            self.metrics['p2_mediated'] = self.dynamics.p2_mediated
            self.metrics['Ns'] = self.dynamics.Ns
            self.metrics['Nf'] = self.dynamics.Nf
            self.metrics['intervention_time'] = self.intervention_time
            self.metrics['node_step'] = self.node_step
            self.metrics['total_step'] = self.total_step
            self.metrics['node_removal'] = self.dynamics.node_removal
            self.metrics['seed'] = self.seed
        
        if len(self.intervention_time) == 0:
            self.metrics['rec_how'] = None
            self.metrics['rec_sample_size'] = None
            self.metrics['rec_distance'] = None
            self.metrics['acc_how'] = None
            self.metrics['acc_prob'] = None
            self.metrics['edge_removal'] = None

        elif time_step == self.intervention_time[0] + 1:
            self.metrics['rec_how'] = self.dynamics.rec_how
            self.metrics['rec_sample_size'] = self.dynamics.rec_sample_size
            self.metrics['rec_distance'] = self.dynamics.rec_distance
            self.metrics['acc_how'] = self.dynamics.acc_how
            self.metrics['acc_prob'] = self.dynamics.rec_acc_prob
            if isinstance(self.metrics['acc_prob'], np.ndarray):
                self.metrics['acc_prob'] = self.metrics['acc_prob'].tolist()
            self.metrics['edge_removal'] = self.dynamics.edge_removal
        
        if time_step % freq == 0:
            self.metrics['time'].append(self.dynamics.network.time_step)
            self.metrics['added_nodes'].append(self.latest_idx)
            if compute_nodes:
                self.metrics['nodes'].append(self.dynamics.network.nodes)
            if compute_edges:
                self.metrics['edges'].append(self.dynamics.network.edges) 
                phases = ['phase0', 'phase1', 'phase2_unmediated', 'phase2_mediated', 'phase3']
                phase_key = ['init', 'ngp1', 'ngp2_unmediated', 'ngp2_mediated', 'rec']
                for i, (p, k) in enumerate(zip(phases, phase_key)):
                    self.metrics[p].append(self.dynamics.network.edge_phase[k])
                    mono, bi = self.dynamics.network.mono_bi_edge_phase(k)
                    self.metrics[f"{p}_mono"].append(mono)
                    self.metrics[f"{p}_bi"].append(bi)
            if compute_degree:
                self.metrics['avg_degree'].append(self.dynamics.network.avg_degree())
            if compute_degree_var:
                self.metrics['degree_variance'].append(self.dynamics.network.degree_var())
            if compute_bi_frac:
                self.metrics['bi_frac'].append(self.dynamics.network.bi_frac)
            if compute_cluster:
                self.metrics['global_clustering'].append(self.dynamics.network.global_clustering())
            if compute_age:
                self.metrics['avg_age'].append(self.dynamics.network.avg_age)
            if compute_nn:
                self.metrics['avg_nn'].append(self.dynamics.network.avg_nodes_dist2)
            if compute_gini:
                self.metrics['gini_coeff'].append(self.dynamics.network.gini_coeff())
            if compute_num_rec:
                self.metrics['num_rec'].append(self.dynamics.num_rec)
            if compute_num_acc:
                self.metrics['num_acc'].append(self.dynamics.num_acc)

            for i in range(len(self.dynamics.grp_lst)):
                grp = self.dynamics.grp_lst[i]
                if compute_nodes:
                    self.metrics[f"nodes_{i}"].append(grp.size)
                if compute_homophily:
                    self.metrics[f"homophily_{i}"].append(self.dynamics.network.homophily(grp))
                if compute_grp_metrics:
                    self.metrics[f"avg_degree_{i}"].append(self.dynamics.network.avg_degree(grp=grp))
                    self.metrics[f"degree_variance_{i}"].append(self.dynamics.network.degree_var(grp=grp))
                    self.metrics[f"global_clustering_{i}"].append(self.dynamics.network.global_clustering(grp=grp))
                    self.metrics[f"gini_coeff_{i}"].append(self.dynamics.network.gini_coeff(grp=grp))

class AB_Experiment(Experiment):
    def __init__(self, grp_info: 'list[dict]', treatment_probability, **kwargs):
        super().__init__(grp_info, **kwargs)
        self.treatment_nodes = []
        # out of the initialized nodes: sample treatment_probability * total nodes to be treatment
        # this need to modify the existingting network at time step 0
        # thouglout the dynamics; whenever a new node is added, toss a coin with probability treatment_probability
        # and assign the node to the treatment group
        
        # two options here: fixed number of treatment nodes or fixed proportion of treatment nodes
        
    def run(self, **kwargs):
        self.intervention_time = kwargs.get('intervention_time', list(range(self.total_step)))
        self.compute_metrics(0, initialize=True, **kwargs)
        self.dynamics.triadic_closure_init()
        for t in range(1, self.total_step+1):
            if t in self.intervention_time:
                step_intervention = True
                # maybe modify step function to take in a list of nodes to be treated or a probability to be treated
                # once the step completed look the new node and externally assign it to treatment group
            else:
                step_intervention = False
            self.latest_idx = self.dynamics.step(self.node_step, self.latest_idx, step_intervention, **kwargs)
            self.compute_metrics(t, **kwargs)
        
        

"""
Conclusion class takes in a list of seeds to run a set of experiments. 

It keeps record of the graph metrics for each experiment, save them into a json file, and calculates the averages & confidence intervals for each metric.
"""
class Conclusion:
    def __init__(self, seeds: 'list[int]'):
        self.seeds = seeds
    
    def run_experiments(self, grp_info: 'list[dict]', **kwargs): 
        self.experiments = []
        record_each_run = kwargs.get('record_each_run', True)
        plot = kwargs.get('plot', False)
        if plot or (not record_each_run):
            self.grp_num = len(grp_info)
            self.total_step = kwargs.get('total_step', 200)

            self.initialize_metrics(**kwargs)
        for i in self.seeds:
            experiment = Experiment(grp_info, seed=i, **kwargs)
            # experiment.dynamics.network.plot_graph()
            experiment.run(**kwargs)
            # experiment.dynamics.network.plot_graph()
            self.experiments.append(experiment)
            if record_each_run:
                if not os.path.isfile('result.json'):
                    with open("result.json", mode='w') as f:
                        json.dump([experiment.metrics], f)
                else:
                    with open("result.json", "r") as f:
                        data = json.loads(f.read())
                    with open("result.json", "w") as f:
                        data.append(experiment.metrics)
                        json.dump(data, f)
                if plot:
                    self.add_metrics(experiment)
            else:
                self.add_metrics(experiment)
        if plot or (not record_each_run):
            self.take_average()
        if plot:
            self.plot_avg_metrics()
    
    def initialize_metrics(self, **kwargs):
        self.compute_nodes = kwargs.get('comp_nodes', True)
        self.compute_edges = kwargs.get('comp_edges', True)
        self.compute_degree = kwargs.get('comp_degree', True)
        self.compute_degree_var = kwargs.get('comp_degvar', True)
        self.compute_bi_frac = kwargs.get('comp_bifrac', True)
        self.compute_cluster = kwargs.get('comp_cluster', True)
        self.compute_age = kwargs.get('comp_age', True)
        self.compute_nn = kwargs.get('comp_nn', True)
        self.compute_homophily = kwargs.get('comp_homop', True)
        self.compute_gini = kwargs.get('comp_gini', True)
        self.compute_num_rec = kwargs.get('comp_num_rec', True)
        self.compute_num_acc = kwargs.get('comp_num_acc', True)
        self.compute_grp_metrics = kwargs.get('comp_grp_metrics', False)
        
        self.avg_metrics = defaultdict(list)
        self.avg_metrics['time'] = []
        self.avg_metrics['added_nodes'] = []

        if self.compute_nodes:
            self.avg_metrics['nodes'] = []
            for i in range(self.grp_num):
                self.avg_metrics[f"nodes_{i}"] = []
        if self.compute_edges:
            self.avg_metrics['edges'] = []
            phases = ['phase0', 'phase1', 'phase2_unmediated', 'phase2_mediated', 'phase3']
            for p in phases:
                self.avg_metrics[p] = []
                self.avg_metrics[f"{p}_mono"] = []
                self.avg_metrics[f"{p}_bi"] = []
        if self.compute_degree:
            self.avg_metrics['avg_degree'] = []
        if self.compute_degree_var:
            self.avg_metrics['degree_variance'] = []
        if self.compute_bi_frac:
            self.avg_metrics['bi_frac'] = []
        if self.compute_cluster:
            self.avg_metrics['global_clustering'] = []
        if self.compute_age:
            self.avg_metrics['avg_age'] = []
        if self.compute_nn:
            self.avg_metrics['avg_nn'] = []
        if self.compute_homophily:
            for i in range(self.grp_num):
                self.avg_metrics[f"homophily_{i}"] = []
        if self.compute_gini:
            self.avg_metrics['gini_coeff'] = []
        if self.compute_num_rec:
            self.avg_metrics['num_rec'] = []
        if self.compute_num_acc:
            self.avg_metrics['num_acc'] = []
        if self.compute_grp_metrics:
            for i in range(self.grp_num):
                self.avg_metrics[f"avg_degree_{i}"] = []
                self.avg_metrics[f"degree_variance_{i}"] = []
                self.avg_metrics[f"global_clustering_{i}"] = []
                self.avg_metrics[f"gini_coeff_{i}"] = []

    def add_metrics(self, experiment: Experiment):
        exp_metrics = experiment.metrics
        self.avg_metrics['time'].append(exp_metrics['time'])
        self.avg_metrics['added_nodes'].append(exp_metrics['added_nodes'])
        if self.compute_nodes:
            self.avg_metrics['nodes'].append(exp_metrics['nodes'])
        if self.compute_edges:
            self.avg_metrics['edges'].append(exp_metrics['edges'])
            phases = ['phase0', 'phase1', 'phase2_unmediated', 'phase2_mediated', 'phase3']
            for p in phases:
                self.avg_metrics[p].append(exp_metrics[p])
                self.avg_metrics[f"{p}_mono"].append(exp_metrics[f"{p}_mono"])
                self.avg_metrics[f"{p}_bi"].append(exp_metrics[f"{p}_bi"])
        if self.compute_degree:
            self.avg_metrics['avg_degree'].append(exp_metrics['avg_degree'])
        if self.compute_degree_var:
            self.avg_metrics['degree_variance'].append(exp_metrics['degree_variance'])
        if self.compute_bi_frac:
            self.avg_metrics['bi_frac'].append(exp_metrics['bi_frac'])
        if self.compute_cluster:
            self.avg_metrics['global_clustering'].append(exp_metrics['global_clustering'])
        if self.compute_age:
            self.avg_metrics['avg_age'].append(exp_metrics['avg_age'])
        if self.compute_nn:
            self.avg_metrics['avg_nn'].append(exp_metrics['avg_nn'])
        if self.compute_gini:
            self.avg_metrics['gini_coeff'].append(exp_metrics['gini_coeff'])
        if self.compute_num_rec:
            self.avg_metrics['num_rec'].append(exp_metrics['num_rec'])
        if self.compute_num_acc:
            self.avg_metrics['num_acc'].append(exp_metrics['num_acc'])
        for i in range(self.grp_num):
            if self.compute_nodes:
                self.avg_metrics[f"nodes_{i}"].append(exp_metrics[f"nodes_{i}"])
            if self.compute_homophily:
                self.avg_metrics[f"homophily_{i}"].append(exp_metrics[f"homophily_{i}"])
            if self.compute_grp_metrics:
                self.avg_metrics[f"avg_degree_{i}"].append(exp_metrics[f"avg_degree_{i}"])
                self.avg_metrics[f"degree_variance_{i}"].append(exp_metrics[f"degree_variance_{i}"])
                self.avg_metrics[f"global_clustering_{i}"].append(exp_metrics[f"global_clustering_{i}"])
                self.avg_metrics[f"gini_coeff_{i}"].append(exp_metrics[f"gini_coeff_{i}"])
    
    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        m = np.average(a, axis=0)
        se = sp.stats.sem(a, axis = 0)
        h = se * sp.stats.t.ppf((1 + confidence) / 2., len(self.seeds)-1)
        return m.tolist(), h.tolist()
    
    def add_stats(self, metric_name):
        self.metric_names.append(metric_name)
        avg, ci = self.mean_confidence_interval(self.avg_metrics[metric_name])
        self.avg_values[metric_name] = avg
        self.ci_values[metric_name] = ci

    def take_average(self):
        self.metric_names, self.avg_values, self.ci_values = [], {}, {}
        self.add_stats('time')
        self.add_stats('added_nodes')
        if self.compute_nodes:
            self.add_stats('nodes')
        if self.compute_edges:
            self.add_stats('edges')
            phases = ['phase0', 'phase1', 'phase2_unmediated', 'phase2_mediated', 'phase3']
            for p in phases:
                self.add_stats(p)
                self.add_stats(f"{p}_mono")
                self.add_stats(f"{p}_bi")
        if self.compute_degree:
            self.add_stats('avg_degree')
        if self.compute_degree_var:
            self.add_stats('degree_variance')
        if self.compute_bi_frac:
            self.add_stats('bi_frac')
        if self.compute_cluster:
            self.add_stats('global_clustering')
        if self.compute_age:
            self.add_stats('avg_age')
        if self.compute_nn:
            self.add_stats('avg_nn')
        if self.compute_gini:
            self.add_stats('gini_coeff')
        if self.compute_num_rec:
            self.add_stats('num_rec')
        if self.compute_num_acc:
            self.add_stats('num_acc')
        for i in range(self.grp_num):
            if self.compute_nodes:
                self.add_stats(f"nodes_{i}")
            if self.compute_homophily:
                self.add_stats(f"homophily_{i}")
            if self.compute_grp_metrics:
                self.add_stats(f"avg_degree_{i}")
                self.add_stats(f"degree_variance_{i}")
                self.add_stats(f"global_clustering_{i}")
                self.add_stats(f"gini_coeff_{i}")
    
    def plot_avg_metrics(self):
        fig, axs = plt.subplots(4, 4, figsize=(25, 20))
        x = self.avg_values['time']
        plt_metric_names = ['added_nodes', 'nodes', 'edges', 'avg_degree', 'degree_variance', 'bi_frac', 'global_clustering', 'avg_age', 'avg_nn', 'gini_coeff', 'num_rec', 'num_acc', 'homophily_0', 'homophily_1']
        for i in range(len(plt_metric_names)):
            avg = np.array(self.avg_values[plt_metric_names[i]])
            ci = np.array(self.ci_values[plt_metric_names[i]])
            axs[i//4, i%4].plot(x, avg)
            axs[i//4, i%4].fill_between(x, (avg + ci), (avg - ci), color='b', alpha=0.1)
            axs[i//4, i%4].set(xlabel='time', ylabel=plt_metric_names[i])
        
        plt_phase_names = ['phase0', 'phase1', 'phase2_unmediated', 'phase2_mediated', 'phase3']
        axs[3, 2].stackplot(x, self.avg_values[plt_phase_names[0]], self.avg_values[plt_phase_names[1]], self.avg_values[plt_phase_names[2]], self.avg_values[plt_phase_names[3]], self.avg_values[plt_phase_names[4]], baseline ='zero', colors =['blue', 'green', 'yellow','orange', 'red'], labels=[plt_phase_names[0], plt_phase_names[1], plt_phase_names[2], plt_phase_names[3], plt_phase_names[4]])
        axs[3, 2].set(xlabel='time', ylabel='edges (phased)')  
        axs[3, 2].legend(loc='best')

        plt_grp_names = ['nodes_0', 'nodes_1']
        axs[3, 3].stackplot(x, self.avg_values[plt_grp_names[0]], self.avg_values[plt_grp_names[1]], baseline ='zero', colors =['blue', 'yellow'], labels=[plt_grp_names[0], plt_grp_names[1]])
        axs[3, 3].set(xlabel='time', ylabel='nodes (grouped)') 
        axs[3, 3].legend(loc='best')
        plt.savefig('result.png') 
        # plt.show()
    
'''
To run an experiment, there are 3 steps:

1. Specify grp_info, a list of map. Each map represents a group, with the following keys: mean, variance, size.

2. Initilize a Conclusion object with a list of seeds, e.g. conclusion = Conclusion(list(range(5)))

3. run conclusion.run_experiments to run a set of experiments.

    Conclusion.run_experiments takes in grp_info and all the optional experiment settings. 
    The following attributes can be specified: (they are annotated with their default values)
    sigmoid coeff: a=2, b=5

    edge initialization: 
        init_how='color' (option: 'embedding')
        conn_matrix=None

    natural growth: 
        ng_how='color' (for natural growth phase 1, options: 'embedding')
        p2_prob=0.05
        p2_mediated=True (whether to consider node candidates that are at distance 2 via medited edges in phase 2)
        Ns=None (number of samples for natural growth phase 1)
        Nf=None (number of samples for natural growth phase 2)

    whether rec: 
        intervention_time=list(range(self.total_step)) (empty means no intervention)

    how rec: 
        rec_how='random_fof' (option: 'embedding', 'adamic_adar')
        rec_sample_size=None
        rec_sample_fraction=None
        rec_distance=2
        acc_how='embedding' (option: 'constant')
        acc_prob=None

    dynamic setting: 
        node_step=20
        total_step=200
        node_removal=True
        death_func=lambda x: 0.0001*np.exp(0.08*x)
        edge_removal=False

    options for computing each metric (True by default): 
        freq=1
        comp_nodes, comp_edges, comp_degree, comp_degvar, comp_bifrac, comp_cluster, comp_age, comp_nn, comp_homop, comp_gini
        comp_grp_metrics=False (compute avg_degree, degree_var, gini_coeff, clustering for each group)

    options for collecting metrics:
        record_each_run=True
        plot=False
'''
if __name__ == "__main__":
    
    # An example of running the experiments
    mu1 = np.array([0,1])
    mu2 = np.array([1,0])
    N1 = 100
    N2 = 100
    sigma1 = 0.05
    sigma2 = 0.05
    a = 2
    b = 5
    conclusion = Conclusion(list(range(3)))
    grp_info = [{'mean': mu1, 'variance': sigma1*np.identity(2), 'size': N1},
                {'mean': mu2, 'variance': sigma2*np.identity(2), 'size': N2}]
    conclusion.run_experiments(grp_info, plot=True, init_how='embedding', Ns=N1+N2, Nf=100, 
                            node_step=20, total_step=1000, acc_how='embedding',acc_prob=50, a=a, b=b,
                            ng_how='embedding', intervention_time=list(range(0, 500)),
                            rec_how='embedding',
                            freq=5, record_each_run=False, rec_sample_size=200)
