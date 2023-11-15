import heapq
import bisect


class ShortestPathSolver:
    def __init__(self, graph):
        self.graph = graph
        self.node_to_component = {}
        self.components = {}
        self.num_components = 0

        # list of edges: [(node1, node2, key, cost), (node2, node3, key, cost)]
        self.multicut = []
        self.node_remap = {}

        # {node: {node: minimal_path_cost}}
        # TODO: do not initialize over all nodes
        self.node_to_predecessor = {node: {} for node in self.graph.nodes}

        # {node1: {node2: [nodes]}
        # from node1 going to node2 you can reach nodes [nodes]
        # this should better be saved on the node directly

        # TODO: maybe save minimum possible cost path per node for optimization

    def get_components(self):
        return self.components

    def get_node_remap(self):
        for node in self.node_remap:
            initial_node = node
            while node in self.node_remap:
                node = self.node_remap[node]
            self.node_remap[initial_node] = node
        return self.node_remap

    def get_lowest_cost_predecessor(self, node, successor=None):
        minimal_cost_predecessor = None
        minimal_cost = 0
        for predecessor, cost in self.node_to_predecessor[node].items():
            if successor is not None and predecessor == successor:
                continue
            if cost < minimal_cost:
                minimal_cost_predecessor = predecessor
                minimal_cost = cost

        return minimal_cost_predecessor, minimal_cost

    def update_cost(self, node, successor):
        # make this function iteratively for efficiency
        minimal_cost_node, minimal_cost = self.get_lowest_cost_predecessor(successor, node)
        # assumption: edge has weight -1
        self.node_to_predecessor[node].update({successor: minimal_cost - 1})
        for predecessor in self.node_to_predecessor[node]:
            if predecessor == successor or predecessor == node:
                continue
            self.update_cost(predecessor, node)

    def update_component_cost(self, component_id):
        component_nodes = self.components[component_id]
        start_nodes = [n for n in component_nodes if len(self.node_to_predecessor[n]) == 1]

        for start_node in start_nodes:
            node_queue = []
            predecessor = start_node
            second_node = list(self.node_to_predecessor[start_node].keys())[0]
            node_queue.append((second_node, predecessor))

            while len(node_queue) > 0:
                current_node, predecessor = node_queue.pop()
                predecessor_cost = self.get_lowest_cost_predecessor(predecessor, current_node)[1]
                edge_cost = min([data["cost"] for data in self.graph.get_edge_data(current_node, predecessor).values()])
                self.node_to_predecessor[current_node].update({predecessor: predecessor_cost + edge_cost})

                for next_node in self.node_to_predecessor[current_node].keys():
                    if next_node == predecessor:
                        continue
                    node_queue.append((next_node, current_node))

    def handle_self_edges(self):
        # handle edges from and to the same node
        for node1, node2, key, data in list(self.graph.edges(data=True, keys=True)):
            if node1 == node2:
                if data.get("cost") < 0:
                    self.multicut.append(data.get("id"))

                # edges with negative cost are cut any way and edges with positive cost can be safely ignored
                self.node_to_predecessor[node1].pop(node2, None)
                self.graph.remove_edge(node1, node2, key)

    def handle_cycle(self, start_node, end_node):
        cycle, cost = self.find_cycle(start_node, end_node)
        if cost >= 0:
            return False

        # remove cut edges from graph and add them to the multicut
        for cycle_node, edge in cycle:
            self.graph.remove_edge(*edge[:3])
            self.multicut.append(edge[3].get("id"))

            # remove nodes from node_to_predecessor
            for neighbour in self.node_to_predecessor[cycle_node]:
                self.node_to_predecessor[neighbour].pop(cycle_node, None)
            self.node_to_predecessor[cycle_node] = {}

        # delete component
        component_id = self.node_to_component[start_node]
        for component_node in self.components[component_id]:
            self.node_to_component.pop(component_node)
        self.components.pop(component_id)

        # merge nodes
        nodes = [item[0] for item in cycle]
        new_node = self.graph.merge_nodes(nodes)
        self.node_to_predecessor[new_node] = {}
        self.node_remap.update(dict([(node, new_node) for node in nodes]))
        # note: node_to_predecessor is not updated in this state, because it will be in the next iteration

    def find_cycle(self, start_node, end_node):
        """
        Finds a cycle starting from the start node and following predecessors until start node is reached again.
        :param end_node:
        :param start_node:
        :return:
        """

        predecessors = {}

        node_queue = []
        current_node = start_node
        cost = 0

        found = []

        last_edge_data = self.graph.get_edge_data(end_node, start_node)
        last_edge_data = list(filter(lambda x: x[1]["cost"] == -1, last_edge_data.items()))[0]
        last_edge = (end_node, start_node, *last_edge_data)
        found.append(tuple(last_edge[:3]))
        found.append((last_edge[1], last_edge[0], last_edge[2]))

        while current_node is not None:
            for neighbor in self.node_to_predecessor[current_node]:
                # get edges between current node and neighbor
                edges_data = self.graph.get_edge_data(current_node, neighbor)
                # filter out already found edges (to prevent using one edge twice)
                edges_data = list(filter(lambda x: (current_node, neighbor, x[0]) not in found and x[1]["cost"] == -1, edges_data.items()))
                if len(edges_data) == 0:
                    continue

                # use edge with minimum cost
                key, min_edge_data = min(edges_data, key=lambda x: x[1]["cost"])

                total_neighbor_cost = cost + min_edge_data["cost"]
                predecessors[neighbor] = current_node, (current_node, neighbor, key, min_edge_data)

                # mark edge as found
                found.append((current_node, neighbor, key))
                found.append((neighbor, current_node, key))

                if neighbor == end_node:
                    # found cycle
                    predecessor_node, edge = predecessors[end_node]
                    cycle = [(end_node, last_edge)]
                    cost += last_edge[3]["cost"]
                    while predecessor_node != start_node:
                        cycle.append((predecessor_node, edge))
                        predecessor_node, edge = predecessors[predecessor_node]
                    cycle.append((start_node, edge))
                    return cycle, total_neighbor_cost

                node_queue.append((neighbor, total_neighbor_cost))

            current_node, cost = node_queue.pop()

    def initial_setup(self):
        reset = True
        while reset:
            reset = False

            self.handle_self_edges()

            # iterate of all nodes initially
            for node in list(self.graph.nodes):
                if reset:
                    break

                # ignore node if already in a component
                if self.node_to_component.get(node) is not None:
                    continue

                # create new component
                new_component = [node]
                component_id = self.num_components
                self.num_components += 1
                self.components[component_id] = new_component
                self.node_to_component[node] = component_id

                # add all nodes to component that have negative cost
                previous_searched_nodes = {}
                unsearched_nodes = [node]
                while len(unsearched_nodes) > 0:
                    if reset:
                        break
                    unsearched_node = unsearched_nodes.pop()
                    previous_searched_node = previous_searched_nodes.get(unsearched_node, None)

                    for _, new_node, cost in list(self.graph.edges(unsearched_node, data="cost")):
                        if previous_searched_node is not None and new_node == previous_searched_node:
                            continue
                        if cost == -1:
                            # check for cycle
                            if new_node in new_component:
                                self.handle_cycle(new_node, unsearched_node)
                                reset = True
                                break

                            previous_searched_nodes[new_node] = unsearched_node

                            # update costs
                            self.update_cost(unsearched_node, new_node)
                            minimal_cost_predecessor, minimal_cost = self.get_lowest_cost_predecessor(unsearched_node, new_node)
                            self.node_to_predecessor[new_node].update({unsearched_node: minimal_cost - 1})

                            # add node to component
                            new_component.append(new_node)
                            unsearched_nodes.append(new_node)
                            self.node_to_component[new_node] = component_id

    def merge_components(self, swallower, swallowee):
        swallowee_nodes = self.components.pop(swallowee)
        self.components[swallower].extend(swallowee_nodes)
        self.node_to_component.update({node: swallower for node in swallowee_nodes})

    def search_from(self, start_node):
        start_component_id = self.node_to_component[start_node]

        found_nodes = {start_node: (None, 0)}
        # (node, predecessor, cost of path until start node)
        node_queue = []
        for neighbor in [n for n in self.graph.neighbors(start_node) if self.node_to_component[n] != start_component_id]:
            neighbor_cost = min([data["cost"] for data in self.graph.get_edge_data(neighbor, start_node).values()])
            bisect.insort(node_queue, (neighbor, start_node, neighbor_cost), key=lambda x: -x[2])

        # TODO: compare found nodes from other components for minimum cost

        while len(node_queue) > 0:
            node, predecessor, cost = node_queue.pop()

            found_nodes[node] = (predecessor, cost)

            lowest_path_cost = self.get_lowest_cost_predecessor(node)[1]
            if lowest_path_cost < 0 and node != start_node:

                component_id = self.node_to_component[node]
                if component_id == start_component_id:
                    # TODO: implement general cycle handling
                    # found cycle
                    continue

                # update node_to_predecessor to prepare for merging
                iter_node = node
                iter_predecessor = predecessor
                while iter_node != start_node:
                    self.node_to_predecessor[iter_node].update({iter_predecessor: 0})
                    self.node_to_predecessor[iter_predecessor].update({iter_node: 0})
                    self.merge_components(start_component_id, self.node_to_component[iter_node])
                    # self.components[start_component_id].append(iter_node)
                    # self.node_to_component[iter_node] = start_component_id
                    iter_node, iter_predecessor = iter_predecessor, found_nodes[iter_predecessor][0]

                self.update_component_cost(start_component_id)

                print(f"merged nodes {start_node} and {node}")

                return True

                # TODO: make condition to continue variably
                if lowest_path_cost < -cost:
                    # cancel search
                    pass
                else:
                    # update found_nodes and node_queue and continue with search
                    pass

            # add next nodes to node queue and calculate cost
            for neighbor in [n for n in self.graph.neighbors(node) if n != predecessor]:
                neighbor_cost = cost + min([data["cost"] for data in self.graph.get_edge_data(neighbor, node).values()])
                if neighbor in found_nodes:
                    if found_nodes[neighbor][1] > neighbor_cost:
                        found_nodes[neighbor] = (node, neighbor_cost)
                    else:
                        continue

                bisect.insort(node_queue, (neighbor, node, neighbor_cost), key=lambda x: -x[2])

        return False

    def search(self):
        ignore_nodes = []
        while len(self.components) > 1:
            min_cost = float("infinity")
            min_nodes = []
            for node in self.graph.nodes:
                if node in ignore_nodes:
                    continue
                cost = self.get_lowest_cost_predecessor(node)[1]
                if cost < min_cost:
                    min_cost = cost
                    min_nodes = []
                if cost <= min_cost:
                    min_nodes.append(node)

            if min_cost >= 0:
                return self.multicut

            self.handle_self_edges()
            start_node = min_nodes[0]
            if not self.search_from(start_node):
                # in this case searching didn't find a new component to merge and no cycle
                ignore_nodes.append(start_node)
                continue

    def solve(self):
        self.initial_setup()

        self.search()

        # extend components:
        # extend node with the lowest cost path (there must be at least two because a path has two ends)
        # add all directly connected nodes
        # if node is already in another component merge components (scores)
        # update path predecessor and score for every node

        # components are necessary to register cycles and how far has been calculated
        # just save predecessor and score

        return self.multicut
