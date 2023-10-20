import pulp


def multicut_ilp(graph):
    # Define the ILP model which is a minimum multicut problem
    model = pulp.LpProblem("Multicut", pulp.LpMinimize)

    # Generate symmetric edges
    edges = list(graph.edges)
    edges += [(v, u) for (u, v) in graph.edges]

    # Create a binary variable for each edge where
    # x(u, v) = 0 means the edge between u and v is not cut
    # x(u, v) = 1 means the edge between u and v is cut
    x = pulp.LpVariable.dicts("x", edges, lowBound=0, upBound=1, cat=pulp.LpInteger)

    # Generate symmetric and non-symmetric vertex pairs
    all_vertex_perm = [pair for pair in pulp.allpermutations(graph.nodes, 2) if len(pair) == 2]
    all_vertex_comb = [pair for pair in pulp.allcombinations(graph.nodes, 2) if len(pair) == 2]

    # Create a binary variable for each vertex pair where
    # path(u, v) = 0 means there is a path from u to v
    # path(u, v) = 1 means there is no path from u to v
    path = pulp.LpVariable.dicts("path", all_vertex_perm, lowBound=0, upBound=1, cat=pulp.LpInteger)

    # Create a PuLP model
    model += pulp.lpSum([graph[u][v]['weight'] * x[(u, v)] for (u, v) in graph.edges])

    # Add the constraints
    for (u, v) in graph.edges:
        model += path[(u, v)] + x[(u, v)] == 1
    for (u, v) in all_vertex_comb:
        for (w, l) in edges:
            if w == u and l != v:
                model += 1 - path[(u, v)] + x[(u, l)] + path[(l, v)] >= 1
                model += 1 - path[(u, v)] + 1 - x[(u, l)] + 1 - path[(l, v)] >= 1
                model += path[(u, v)] + x[(u, l)] + 1 - path[(l, v)] >= 1

    # Solve the problem
    model.solve()

    # Print the results for debugging
    # print("Status: ", pulp.LpStatus[model.status])
    # print("Optimal value: ", pulp.value(model.objective))
    # for e in graph.edges:
    #     print("Edge ({},{}): {}".format(e[0], e[1], pulp.value(x[e])))

    # Extract the optimal solution and return
    opt_cut = []
    for (u, v) in graph.edges():
        if x[(u, v)].varValue == 1:
            opt_cut.append((u, v))
            opt_cut.append((v, u))
    return opt_cut, pulp.value(model.objective)
