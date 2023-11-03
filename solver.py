from graph import Graph


def shortest_path_solve(graph: Graph):
    # initialize components

    node_to_component = {}
    components = {}
    num_components = 0

    for node in graph.nodes:
        if node_to_component.get(node) is not None:
            continue

        new_component = [node]
        component_id = num_components
        num_components += 1
        components[component_id] = new_component
        node_to_component[node] = component_id
        unsearched_nodes = [node]

        while len(unsearched_nodes) > 0:
            unsearched_node = unsearched_nodes.pop()
            edges = graph.edges(unsearched_node, data="cost")
            for edge in edges:
                if edge[2] == -1:
                    if edge[0] == unsearched_node:
                        new_node = edge[1]
                    else:
                        new_node = edge[0]
                    if new_node in new_component:
                        continue
                    new_component.append(new_node)
                    unsearched_nodes.append(new_node)
                    node_to_component[new_node] = new_component

    return components
