from graph import Graph


class ShortestPathSolver:
    def __init__(self, graph):
        self.graph = graph
        self.node_to_component = {}
        self.components = {}
        self.num_components = 0

        # {node: {node: minimal_path_cost}}
        self.node_to_predecessor = {node: {} for node in self.graph.nodes}

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
            if predecessor == successor:
                continue
            self.update_cost(predecessor, node)

    def solve(self):
        for node in self.graph.nodes:
            if self.node_to_component.get(node) is not None:
                continue

            new_component = [node]
            component_id = self.num_components
            self.num_components += 1
            self.components[component_id] = new_component
            self.node_to_component[node] = component_id
            unsearched_nodes = [node]

            while len(unsearched_nodes) > 0:
                unsearched_node = unsearched_nodes.pop()
                edges = self.graph.edges(unsearched_node, data="cost")

                for edge in edges:
                    if edge[2] == -1:
                        if edge[0] == unsearched_node:
                            new_node = edge[1]
                        else:
                            new_node = edge[0]
                        if new_node in new_component:
                            continue

                        self.update_cost(unsearched_node, new_node)
                        minimal_cost_predecessor, minimal_cost = self.get_lowest_cost_predecessor(unsearched_node, new_node)
                        self.node_to_predecessor[new_node].update({unsearched_node: minimal_cost - 1})

                        new_component.append(new_node)
                        unsearched_nodes.append(new_node)
                        self.node_to_component[new_node] = new_component

        # find cycles (not yet implemented; implement in the above algorithm)

        # extend components:
        # extend node with the lowest cost path (there must be at least two because a path has two ends)
        # add all directly connected nodes
        # if node is already in another component merge components (scores)
        # update path predecessor and score for every node

        # components are necessary to register cycles and how far has been calculated
        # just save predecessor and score

        return self.components
