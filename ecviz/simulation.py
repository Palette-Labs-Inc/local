import networkx as nx
import numpy as np
import pandas as pd

class Simulation:
    def __init__(self, n1, p1, n2, p2, weight_selection, weight_scaling, which_edge_create, alpha_ec, ec_compute_selection, use_weight_compute_ec):
        self.n1 = n1
        self.p1 = p1
        self.n2 = n2
        self.p2 = p2
        self.weight_selection = weight_selection
        self.weight_scaling = weight_scaling
        self.which_edge_create = which_edge_create
        self.alpha_ec = alpha_ec
        self.alpha_w = 1-self.alpha_ec
        self.ec_compute_selection = ec_compute_selection
        self.use_weight_compute_ec = use_weight_compute_ec

        self.rng = np.random.default_rng()
        self.step = 0
        self.G, self.G1, self.G2 = self.create_initial_clusters(n1, p1, n2, p2)
        self.pos = nx.spring_layout(self.G)

        self.centrality_data_cluster1 = []
        self.centrality_data_cluster2 = []
        self.centrality_sums = []
        self.weight_sum_G1 = []
        self.weight_sum_G2 = []
        self.weight_sums = []
        self.graph_value_cluster1 = []
        self.graph_value_cluster2 = []
        self.graph_value_sums = []

    def update_alpha_ec(self, alpha_ec):
        print('updating alpha')
        self.alpha_ec = alpha_ec
        self.alpha_w = 1 - alpha_ec

    def get_weight(self):
        if self.weight_selection == "U[0,1)":
            return self.rng.uniform()
        elif self.weight_selection == "1":
            return 1

    def create_initial_clusters(self, n1, p1, n2, p2):
        seed = np.random.randint(1, 10000)
        G1 = nx.erdos_renyi_graph(n1, p1, seed=seed+1)
        G2 = nx.erdos_renyi_graph(n2, p2, seed=seed+2)

        for (u, v) in G1.edges():
            G1[u][v]['weight'] = self.get_weight()
        for (u, v) in G2.edges():
            G2[u][v]['weight'] = self.get_weight()

        G = nx.disjoint_union_all([G1, G2])
        return G, G1, G2

    def update_weight(self, G, G1, G2, graph_id, u, v, new_weight):
        if graph_id == 1:
            if G1.has_edge(u, v):
                G1[u][v]['weight'] += new_weight
            else:
                G1.add_edge(u, v, weight=new_weight)
            if G.has_edge(u, v):
                G[u][v]['weight'] += new_weight
            else:
                G.add_edge(u, v, weight=new_weight)
        elif graph_id == 2:
            if G2.has_edge(u, v):
                G2[u][v]['weight'] += new_weight
            else:
                G2.add_edge(u, v, weight=new_weight)
            offset = max(G1.nodes) + 1
            G_node_u = u + offset
            G_node_v = v + offset
            if G.has_edge(G_node_u, G_node_v):
                G[G_node_u][G_node_v]['weight'] += new_weight
            else:
                G.add_edge(G_node_u, G_node_v, weight=new_weight)
        elif graph_id == 3:
            if G.has_edge(u, v):
                G[u][v]['weight'] += new_weight
            else:
                G.add_edge(u, v, weight=new_weight)

    def calculate_eigenvector_centrality(self, G):
        ec_kwargs = {
            'max_iter': 1000,
            'tol': 1e-2,
        }
        if self.use_weight_compute_ec:
            ec_kwargs['weight'] = 'weight'
        centrality = nx.eigenvector_centrality(G, **ec_kwargs)
        return centrality

    def calculate_node_weight_sums(self, G):
        weight_sums = {node: 0 for node in G.nodes()}
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 0.0)
            weight_sums[u] += weight
            weight_sums[v] += weight
        return weight_sums

    def step_graph_value(self):
        # get previous steps graph value information
        if len(self.graph_value_cluster1) > 0:  # assume the rest are filled
            # get previous weights
            prev_weights_G1 = self.weight_sum_G1[-1]
            prev_weights_G2 = self.weight_sum_G2[-1]
            # get previous EC
            prev_centrality_data_1 = self.centrality_data_cluster1[-1]
            prev_centrality_data_2 = self.centrality_data_cluster2[-1]
            # get previous graph value
            prev_gv_data_1 = self.graph_value_cluster1[-1]
            prev_gv_data_2 = self.graph_value_cluster2[-1]

            # now compute the delta in weight & EC, and update graph value accordingly
            ec_curr = self.calculate_eigenvector_centrality(self.G)
            w_curr = self.calculate_node_weight_sums(self.G)

            node2gv = {}
            # print(self.G.number_of_nodes())
            # print(ec_curr)
            # print(w_curr)
            for node in range(self.G.number_of_nodes()):
                if node < self.G1.number_of_nodes():
                    delta_weight = w_curr[node] - prev_weights_G1[node]
                    delta_ec = ec_curr[node] - prev_centrality_data_1[node]
                    gv = prev_gv_data_1[node] + delta_weight**(1-self.alpha_ec) * delta_ec**self.alpha_ec
                else:
                    delta_weight = w_curr[node] - prev_weights_G2[node - self.G1.number_of_nodes()]
                    delta_ec = ec_curr[node] - prev_centrality_data_2[node - self.G1.number_of_nodes()]
                    gv = prev_gv_data_2[node - self.G1.number_of_nodes()] + delta_weight**(1-self.alpha_ec) * delta_ec**self.alpha_ec
                print(node, delta_weight, delta_ec, gv)
                node2gv[node] = gv

        else:
            # now compute the delta in weight & EC, and update graph value accordingly
            ec_curr = self.calculate_eigenvector_centrality(self.G)
            w_curr = self.calculate_node_weight_sums(self.G)

            node2gv = {}
            for node in range(self.G.number_of_nodes()):
                gv = w_curr[node]**(1-self.alpha_ec) * ec_curr[node]**self.alpha_ec
                node2gv[node] = gv
        # print(node2gv)

        # ec = self.calculate_eigenvector_centrality(G)
        # w = self.calculate_node_weight_sums(G)
        # gv = {node: (self.alpha_ec * ec[node]) * (self.alpha_w * w[node]) for node in G.nodes()}
        # return gv
        return node2gv

    def initialize_simulation(self):
        self.G, self.G1, self.G2 = self.create_initial_clusters(self.n1, self.p1, self.n2, self.p2)
        self.pos = nx.spring_layout(self.G)
        self.step = 0

        centrality = self.calculate_eigenvector_centrality(self.G)
        centrality_data_1 = [centrality[node] for node in range(self.G1.number_of_nodes())]
        centrality_data_2 = [centrality[node] for node in range(self.G1.number_of_nodes(), self.G1.number_of_nodes() + self.G2.number_of_nodes())]
        centrality_sum_1 = sum(centrality[node] for node in range(self.G1.number_of_nodes()))
        centrality_sum_2 = sum(centrality[node] for node in range(self.G1.number_of_nodes(), self.G1.number_of_nodes() + self.G2.number_of_nodes()))
        total_centrality_sum = sum(centrality.values())

        gv = self.step_graph_value()
        gv_data_1 = [gv[node] for node in range(self.G1.number_of_nodes())]
        gv_data_2 = [gv[node] for node in range(self.G1.number_of_nodes(), self.G1.number_of_nodes() + self.G2.number_of_nodes())]
        gv_sum_1 = sum(gv[node] for node in range(self.G1.number_of_nodes()))
        gv_sum_2 = sum(gv[node] for node in range(self.G1.number_of_nodes(), self.G1.number_of_nodes() + self.G2.number_of_nodes()))
        total_gv_sum = sum(gv.values())


        weights_G1 = self.calculate_node_weight_sums(self.G1)
        weights_G2 = self.calculate_node_weight_sums(self.G2)
        weight_sum_1 = sum(weights_G1.values())
        weight_sum_2 = sum(weights_G2.values())
        total_weight_sum = weight_sum_1 + weight_sum_2

        self.centrality_sums.append([0, centrality_sum_1, centrality_sum_2, total_centrality_sum])
        self.centrality_data_cluster1.append(centrality_data_1)
        self.centrality_data_cluster2.append(centrality_data_2)
        self.weight_sum_G1.append(weights_G1)
        self.weight_sum_G2.append(weights_G2)
        self.weight_sums.append([0, weight_sum_1, weight_sum_2, total_weight_sum])
        self.graph_value_cluster1.append(gv_data_1)
        self.graph_value_cluster2.append(gv_data_2)
        self.graph_value_sums.append([0, gv_sum_1, gv_sum_2, total_gv_sum])

    def run_epoch(self):
        self.step += 1

        if np.random.rand() < self.which_edge_create:
            uv = np.random.choice(self.G1.nodes(), size=2, replace=False)
            u = uv[0]
            v = uv[1]
            G1_weight = self.get_weight()
            self.update_weight(self.G, self.G1, self.G2, 1, u, v, G1_weight)
            
            uv = np.random.choice(self.G2.nodes(), size=2, replace=False)
            u = uv[0]
            v = uv[1]
            G2_weight = G1_weight * self.weight_scaling
            self.update_weight(self.G, self.G1, self.G2, 2, u, v, G2_weight)

            # weight_update = (G1_weight, G2_weight)
        else:
            u = np.random.randint(0, self.G1.number_of_nodes())
            v = np.random.randint(self.G1.number_of_nodes(), self.G.number_of_nodes())
            G_weight = self.get_weight()
            self.update_weight(self.G, self.G1, self.G2, 3, u, v, G_weight)

            # weight_update = (G_weight, )

        centrality = self.calculate_eigenvector_centrality(self.G)
        centrality_data_1 = [centrality[node] for node in range(self.G1.number_of_nodes())]
        centrality_data_2 = [centrality[node] for node in range(self.G1.number_of_nodes(), self.G1.number_of_nodes() + self.G2.number_of_nodes())]
        centrality_sum_1 = sum(centrality[node] for node in range(self.G1.number_of_nodes()))
        centrality_sum_2 = sum(centrality[node] for node in range(self.G1.number_of_nodes(), self.G1.number_of_nodes() + self.G2.number_of_nodes()))
        total_centrality_sum = sum(centrality.values())

        gv = self.step_graph_value()
        gv_data_1 = [gv[node] for node in range(self.G1.number_of_nodes())]
        gv_data_2 = [gv[node] for node in range(self.G1.number_of_nodes(), self.G1.number_of_nodes() + self.G2.number_of_nodes())]
        gv_sum_1 = sum(gv[node] for node in range(self.G1.number_of_nodes()))
        gv_sum_2 = sum(gv[node] for node in range(self.G1.number_of_nodes(), self.G1.number_of_nodes() + self.G2.number_of_nodes()))
        total_gv_sum = sum(gv.values())
        
        self.centrality_data_cluster1.append(centrality_data_1)
        self.centrality_data_cluster2.append(centrality_data_2)
        self.centrality_sums.append([self.step, centrality_sum_1, centrality_sum_2, total_centrality_sum])

        self.graph_value_cluster1.append(gv_data_1)
        self.graph_value_cluster2.append(gv_data_2)
        self.graph_value_sums.append([self.step, gv_sum_1, gv_sum_2, total_gv_sum])

        weight_G1 = self.calculate_node_weight_sums(self.G1)
        weight_G2 = self.calculate_node_weight_sums(self.G2)
        weight_sum_1 = sum(weight_G1.values())
        weight_sum_2 = sum(weight_G2.values())
        total_weight_sum = weight_sum_1 + weight_sum_2
        self.weight_sum_G1.append(weight_G1)
        self.weight_sum_G2.append(weight_G2)
        self.weight_sums.append([self.step, weight_sum_1, weight_sum_2, total_weight_sum])
