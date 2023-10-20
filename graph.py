import random

import networkx as nx
import numpy as np


class GraphFactory:
    @staticmethod
    def generate_grid(size):
        # get weights
        num_edges = 2 * size[0] * size[1] - size[0] - size[1]
        num_nodes = size[0] * size[1]
        costs = GraphFactory.get_costs(num_edges)

        # generate vertices
        graph = Graph()

        # generate nodes
        graph.add_nodes_from([(node_id, {'pos': np.array((x, y))}) for node_id, (x, y) in
                              enumerate([(x, y) for x in range(size[0]) for y in range(size[1])])])

        # generate edges
        edges = []
        i = 0
        for node1 in range(num_nodes):
            for node2 in range(node1 + 1, num_nodes):
                if ((graph.nodes[node1]['pos'] - graph.nodes[node2]['pos']) ** 2).sum() == 1:
                    edges.append((node1, node2, {'cost': costs[i]}))
                    i += 1
        graph.add_edges_from(edges)

        return graph

    @staticmethod
    def get_costs(num_edges, seed=None):
        costs = []
        if seed is None:
            for i in range(num_edges):
                costs.append(random.choice([-1, 1]))
        else:
            seed = str(bin(seed))[3:]
            for value in seed:
                costs.append(int(value) if int(value) == 1 else -1)

        return costs


class Graph(nx.Graph):
    pass
