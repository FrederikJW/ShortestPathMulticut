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
                costs.append(random.choices([-1, 1], [0.2, 0.8])[0])
        else:
            seed = str(bin(seed))[3:]
            for value in seed:
                costs.append(int(value) if int(value) == 1 else -1)

        return costs

    @staticmethod
    def generate_grid_search_graph(graph):
        max_x, max_y = tuple(graph.get_max_pos())
        search_graph = Graph()

        # generate nodes
        nodes_map = dict(enumerate([(x, y) for x in range(max_x) for y in range(max_y)]))
        nodes_map_inv = {pos: node_id for node_id, pos in nodes_map.items()}
        search_graph.add_nodes_from([(node_id, {'pos': np.array((x, y))}) for node_id, (x, y) in nodes_map.items()])
        search_graph.add_node(-1, pos=np.array((-1, -1)))
        nodes_map_inv[(-1, -1)] = -1

        edges = []
        num_nodes = len(graph.nodes)
        for node1 in range(num_nodes):
            for node2 in range(node1 + 1, num_nodes):
                node1_pos = graph.nodes[node1]['pos']
                node2_pos = graph.nodes[node2]['pos']
                if ((node1_pos - node2_pos) ** 2).sum() == 1:
                    if node1_pos[0] == 0 and node2_pos[0] == 0:
                        # edge leads to left outside
                        x1 = 0
                        y1 = min(node1_pos[1], node2_pos[1])
                        x2 = -1
                        y2 = -1
                    elif node1_pos[0] == max_x and node2_pos[0] == max_x:
                        # edge leads to right outside
                        x1 = max_x - 1
                        y1 = min(node1_pos[1], node2_pos[1])
                        x2 = -1
                        y2 = -1
                    elif node1_pos[1] == 0 and node2_pos[1] == 0:
                        # edge leads to top outside
                        x1 = min(node1_pos[0], node2_pos[0])
                        y1 = 0
                        x2 = -1
                        y2 = -1
                    elif node1_pos[1] == max_y and node2_pos[1] == max_y:
                        # edge leads to bottom outside
                        x1 = min(node1_pos[0], node2_pos[0])
                        y1 = max_y - 1
                        x2 = -1
                        y2 = -1
                    else:
                        if node1_pos[0] == node2_pos[0]:
                            # edge is inside horizontally
                            x1 = node1_pos[0] - 1
                            y1 = min(node1_pos[1], node2_pos[1])
                            x2 = node1_pos[0]
                            y2 = y1
                        else:
                            # edge is inside vertically
                            x1 = min(node1_pos[0], node2_pos[0])
                            y1 = node1_pos[1] - 1
                            x2 = x1
                            y2 = node1_pos[1]

                    key = 0
                    if search_graph.has_edge(nodes_map_inv[(x1, y1)], nodes_map_inv[(x2, y2)], key=0):
                        key = 1
                    search_graph.add_edge(nodes_map_inv[(x1, y1)], nodes_map_inv[(x2, y2)], key=key,
                                          cost=graph.get_edge_data(node1, node2)[0].get("cost"))

        return search_graph


class Graph(nx.MultiGraph):
    def get_max_pos(self):
        max_x = 0
        max_y = 0
        for node in self.nodes(data="pos"):
            if node[1][0] > max_x:
                max_x = node[1][0]
            if node[1][1] > max_y:
                max_y = node[1][1]

        return np.array((max_x, max_y))

    def load_value(self, node_to_value, key_word):
        attrs = {node: {key_word: node_to_value[node]} for node in self.nodes}
        nx.set_node_attributes(self, attrs)

