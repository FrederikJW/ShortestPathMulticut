import random
import h5py
from tqdm import tqdm

# seed_value = random.randint(1, 1000)
seed_value = 19
random.seed(seed_value)
print(seed_value)

import networkx as nx
import numpy as np


class GraphFactory:
    @staticmethod
    def construct_from_values(nodes, edges):
        graph = Graph()
        graph.add_nodes_from([(node, {"pos": (x, y)}) for node, x, y in nodes])
        graph.add_edges_from([(node1, node2, {"cost": cost, "id": edge_id}) for node1, node2, _, cost, edge_id in edges])

        return graph

    @staticmethod
    def read_slice_from_snemi3d(slice_num, size=None):
        graph = Graph()
        try:
            with h5py.File("graphs/hdf5/SNEMI3Daffinities.hdf", 'r') as file:
                print(f"Loading graph from SNEMI3D Database from slice {slice_num}")
                data = file['vol0'][...]
        except IOError:
            print("File not accessible")
        except KeyError:
            print("Dataset not found in the file")

        num_nodes = data[0][0].size
        col_count = data[0][0].shape[0]
        if size is not None:
            num_nodes = size[0] * size[1]
            col_count = size[0]

        graph.add_nodes_from([(node, {"pos": (node % col_count, node // col_count)}) for node in range(num_nodes)])

        edges = []
        i = 0
        for node in tqdm(range(num_nodes)):
            x = node % col_count
            y = node // col_count

            if x != 0:
                edges.append((node, node - 1, {"cost": (data[1][slice_num][x][y] * 2) - 1, "id": i}))
                i += 1
            if y != 0:
                edges.append((node, node - col_count, {"cost": (data[2][slice_num][x][y] * 2) - 1, "id": i}))
                i += 1

        graph.add_edges_from(edges)

        return graph

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
                              enumerate([(x, y) for y in range(size[1]) for x in range(size[0])])])

        # generate edges
        edges = []
        i = 0
        for node1 in range(num_nodes):
            for node2 in range(node1 + 1, num_nodes):
                if ((graph.nodes[node1]['pos'] - graph.nodes[node2]['pos']) ** 2).sum() == 1:
                    edges.append((node1, node2, {'cost': costs[i], 'id': i}))
                    i += 1
        graph.add_edges_from(edges)

        return graph

    @staticmethod
    def get_costs(num_edges, seed=None):
        costs = []
        if seed is None:
            for i in range(num_edges):
                costs.append(random.choices([-1, 1], [0.3, 0.7])[0])
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
        outside_node_id = search_graph.number_of_nodes()
        search_graph.add_node(outside_node_id, pos=np.array((-1, -1)))
        nodes_map_inv[(-1, -1)] = outside_node_id

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

                    # assumption: the original graph only has one edge for any two nodes u and v
                    data = graph.get_edge_data(node1, node2)[0]
                    search_graph.add_edge(nodes_map_inv[(x1, y1)], nodes_map_inv[(x2, y2)], key=key,
                                          cost=data.get("cost"), id=data.get("id"))

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

    def load_values(self, node_to_value, key_word):
        attrs = {node: {key_word: node_to_value[node]} for node in self.nodes}
        nx.set_node_attributes(self, attrs)

    def load_single_value(self, value, key_word):
        nx.set_node_attributes(self, value, key_word)

    def get_min_cost_edge_data(self, node1, node2):
        return node1, node2, *min([(key, data) for key, data in
                                   self.get_edge_data(node1, node2).items()], key=lambda x: x[1]["cost"])

    def merge_nodes(self, nodes_to_merge):
        new_node = max(self.nodes) + 1

        self.add_node(new_node)

        for node1, node2, attrs in self.edges(nodes_to_merge, data=True):
            node1 = new_node
            node2 = new_node if node2 in nodes_to_merge else node2
            self.add_edge(node1, node2, **attrs)

        self.remove_nodes_from(nodes_to_merge)

        return new_node

    def export(self):
        nodes = [(node[0], node[1][0], node[1][1]) for node in self.nodes(data="pos")]
        edges = [(node1, node2, key, data.get("cost"), data.get("id")) for node1, node2, key, data in
                 self.edges(data=True, keys=True)]
        return nodes, edges

    def standard_export(self):
        nodes = self.number_of_nodes()
        edges = [(node1, node2, data.get("cost"), data.get("id")) for node1, node2, data in self.edges(data=True)]
        edges.sort(key=lambda x: x[3])
        edges = [(node1, node2, cost) for node1, node2, cost, _ in edges]

        return nodes, edges

