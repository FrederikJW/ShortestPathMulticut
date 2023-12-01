#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <map>
#include <vector>
#include <tuple>

// Edge structure to store cost
struct Edge {
    int node1;
    int node2;
    int key;
    int cost;
    int id;

    Edge() : node1(0), node2(0), key(0), cost(0), id(0) {};
    Edge(int n1, int n2, int k, int c, int i) : node1(n1), node2(n2), key(k), cost(c), id(i) {};
};

// Node structure to store color
struct Node {
    int id;
    int posx;
    int posy;

    Node() : id(-1) {};
    Node(int i, int x, int y) : id(i), posx(x), posy(y) {};
};

// Graph class
class Graph {
private:
    std::map<int, Edge> allEdges;
    std::map<int, std::map<int, std::map<int, Edge>>> adjacencyList;
    std::map<int, Node> allNodes;
    
public:
    // Function to add an edge with a specified cost
    void addEdge(int v1, int v2, int key, int cost, int id) {
        adjacencyList[v1][v2][key] = Edge(v1, v2, key, cost, id);
        adjacencyList[v2][v1][key] = Edge(v2, v1, key, cost, id);
        allEdges[id] = Edge(v1, v2, key, cost, id);
    };

    void addNode(int id, int posx, int posy) {
        auto it = allNodes.find(id);
        if (it == allNodes.end()) {
            allNodes[id] = Node(id, posx, posy);
        };
    };

    std::vector<Edge> getAllEdges() {
        std::vector<Edge> edges;
        for (const auto& pair : allEdges) {
            edges.push_back(pair.second);
        }
        return edges;
    }

    std::vector<Edge> getEdges(int node) {
        std::vector<Edge> edges;

        for (const auto& pair1 : adjacencyList[node]) {
            for (const auto& pair2 : pair1.second) {
                edges.push_back(pair2.second)
            }
        }
    }

    std::vector<int> getAllEdgeIds() {
        std::vector<int> edges;
        for (const auto& pair : allEdges) {
            edges.push_back(pair.first);
        }
        return edges;
    }

    std::vector<Node> getAllNodes() {
        std::vector<Node> nodes;
        for (const auto& pair : allNodes) {
            nodes.push_back(pair.second);
        }
        return nodes;
    }

    std::vector<int> getAllNodeIds() {
        std::vector<int> nodes;
        for (const auto& pair : allNodes) {
            nodes.push_back(pair.first);
        }
        return nodes;
    }

    std::tuple<int, int> maxPos() {
        int maxx = 0;
        int maxy = 0;
        for (const auto& node : allNodes) {
            if (node.second.posx > maxx) {
                maxx = node.second.posx;
            }
            if (node.second.posy > maxy) {
                maxy = node.second.posy;
            }
        }
        std::tuple<int, int> maxPos(maxx, maxy);
        return maxPos;
    }
};

class Solver {
private:
    Graph graph;
    Graph searchGraph;
    std::map<int, int> nodeToComponent;
    std::map<int, std::vector<int>> components;
    std::map<int, std::map<int, int>> nodeToPredecessor;


public:
    void loadGraph(const std::vector<std::tuple<int, int, int>>& nodes, const std::vector<std::tuple<int, int, int, int, int>>& edges) {
        for (const auto& node : nodes) {
            graph.addNode(std::get<0>(node), std::get<1>(node), std::get<2>(node));
        }

        for (const auto& edge : edges) {
            graph.addEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge), std::get<3>(edge), std::get<4>(edge));
        }

        buildSearchGraph();
    };

    void loadSearchGraph(const std::vector<std::tuple<int, int, int>>& nodes, const std::vector<std::tuple<int, int, int, int, int>>& edges) {
        for (const auto& node : nodes) {
            searchGraph.addNode(std::get<0>(node), std::get<1>(node), std::get<2>(node));
        }

        for (const auto& edge : edges) {
            searchGraph.addEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge), std::get<3>(edge), std::get<4>(edge));
        }
    };

    void buildSearchGraph() {
        std::tuple<int, int> maxPos = graph.maxPos();
        int maxx = std::get<0>(maxPos);
        int maxy = std::get<1>(maxPos);
        
        int n = 0;
        for (size_t x = 0; x < maxx; x++) {
            for (size_t y = 0; y < maxy; y++) {
                searchGraph.addNode(n, x, y);
                n++;
            }
        }
        searchGraph.addNode(n, -1, -1);

        // implementation not finished
    };

    int getNewComponentId() {
        auto it = components.rbegin();
        int greatestKey = (it != components.rend()) ? it->first : 0;
        return greatestKey + 1;
    };

    void handleSelfEdges() {
        // implementation not finished
        ;
    };

    std::tuple<std::vector<std::tuple<int, Edge>>, int> findPath(int startNode, int endNode) {
        if (startNode == endNode) {
            std::vector < std::tuple<int, Edge> emptyPath;
            return std::tuple<std::vector<std::tuple<int, Edge>>, int> result(emptyPath, 0);
        };

        std::map<int, Edge> predecessors;
        std::vector<int> nodeQueue;
        auto currentNode = startNode;
        int cost = 0;
        std::vector<int> foundEdges;
        std::vector<int> foundNodes;

        while (currentNode != nodeQueue.end()) {
            for (const auto& pair : nodeToPredecessor[currentNode]) {
                if (std::binary_search(foundNodes.begin(), foundNodes.end(), pair.first)) continue;

            };
        };
    };

    void initialSetup() {
        bool reset = true;

        while (reset) {
            reset = false;

            // iterate over all nodes initially
            for (int nodeId : searchGraph.getAllNodeIds()) {
                if (reset) break;

                // ignore node if already in a component
                auto component = nodeToComponent.find(nodeId);
                if (component != nodeToComponent.end()) {
                    continue;
                };

                // create new component
                std::vector<int> newComponent;
                newComponent.push_back(nodeId);
                int newComponentId = getNewComponentId();
                components[newComponentId] = newComponent;
                nodeToComponent[nodeId] = newComponentId;

                // add all nodes to component that have negative cost
                std::map<int, int> previousSearchedNodes;
                std::vector<int> unsearchdNodes;
                while (!unsearchdNodes.empty()) {
                    if (reset) break;
                    int unsearchedNode = unsearchdNodes.back();
                    unsearchdNodes.pop_back();
                    auto previousSearchedNode = previousSearchedNodes.find(unsearchedNode);

                    for (const Edge edge : graph.getEdges(unsearchedNode)) {
                        if (previousSearchedNode == edge.node2) continue;

                        if (edge.cost == -1) {
                            // check for cycle
                            if (std::binary_search(newComponent.begin(), newComponent.end(), edge.node2)) {
                                // cycle found
                                // requires findCylce and handleCycle

                                reset = true;
                                break;
                            }

                            previousSearchedNodes[edge.node2] = unsearchedNode;

                            // update costs
                            // requires updateCost

                            // add node to component
                            auto it = std::lower_bound(newComponent.begin(), newComponent.end(), edge.node2);
                            orderedVector.insert(it, edge.node2);
                            unsearchdNodes.push_back(edge.node2);
                            nodeToComponent[edge.node2] = newComponentId;
                        }
                    }
                }
            };
        };
    };

    void solve() {
        initialSetup();
    };
};

Solver getSolver() {
    return Solver();
};

int main() {
    // Create a graph
    Graph g;

    // Add edges with weights
    g.addEdge(0, 1, 1, 3, 1);
    g.addEdge(0, 2, 1, 5, 2);
    g.addEdge(1, 2, 1, 2, 3);
    g.addEdge(1, 3, 1, 7, 4);
    g.addEdge(0, 1, 2, 2, 5);
    g.addEdge(1, 0, 3, 3, 6);


    // Print the graph with colors
    std::cout << "Finished\n";

    return 0;
}


namespace py = pybind11;

PYBIND11_MODULE(spm_solver, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: spm_solver

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";
    
    pybind11::class_<Solver>(m, "Solver")
        .def(pybind11::init<>())
        .def("load_graph", &Solver::loadGraph)
        .def("load_search_graph", &Solver::loadSearchGraph)
        .def("solve", &Solver::solve);

    m.def("get_solver", &getSolver, R"pbdoc(
        Creates a Solver which handles solving the multicut problem.
    )pbdoc");

}
