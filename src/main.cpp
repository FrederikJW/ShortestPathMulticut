#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <algorithm>
#include <optional>
#include <map>
#include <vector>
#include <tuple>
#include <queue>
#include <stdexcept>
#include <limits>
#include <mutex>
#include <thread>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace py = pybind11;

const bool debug = false;

const double MAX_DOUBLE = std::numeric_limits<double>::max();

struct TupleComparator {
    bool operator()(const std::tuple<int, int, double>& lhs, const std::tuple<int, int, double>& rhs) const {
        return std::get<2>(lhs) < std::get<2>(rhs);
    }
};

// Edge structure to store cost
struct Edge {
    int node1;
    int node2;
    int key;
    double cost;
    int id;

    Edge() : node1(0), node2(0), key(0), cost(0), id(0) {};
    Edge(int n1, int n2, int k, double c, int i) : node1(n1), node2(n2), key(k), cost(c), id(i) {};
};

// Node structure to store color
struct Node {
    int id;
    int posx;
    int posy;

    Node() : id(-1), posx(-1), posy(-1) {};
    Node(int i, int x, int y) : id(i), posx(x), posy(y) {};
};

// Graph class
class Graph {
private:
    std::map<int, Edge> allEdges;
    std::map<int, std::map<int, std::map<int, Edge>>> adjacencyList;
    std::map<int, Node> allNodes;
    int maxNodeId;
    
public:
    Graph() {
        maxNodeId = 0;
    }

    int getMaxNodeId() {
        return maxNodeId;
    };

    // Function to add an edge with a specified cost
    void addEdge(int v1, int v2, int key, double cost, int id) {
        adjacencyList[v1][v2][key] = Edge(v1, v2, key, cost, id);
        adjacencyList[v2][v1][key] = Edge(v2, v1, key, cost, id);
        allEdges[id] = Edge(v1, v2, key, cost, id);
    };

    void addEdge(int v1, int v2, double cost, int id) {
        auto keyToEdge = adjacencyList[v1][v2];
        int newKey = 0;
        for (const auto& pair : keyToEdge) {
            if (pair.first > newKey) newKey = pair.first;
        };
        newKey++;

        adjacencyList[v1][v2][newKey] = Edge(v1, v2, newKey, cost, id);
        adjacencyList[v2][v1][newKey] = Edge(v2, v1, newKey, cost, id);
        allEdges[id] = Edge(v1, v2, newKey, cost, id);
    };

    void removeEdge(int v1, int v2, int key, int id) {
        adjacencyList[v1][v2].erase(key);
        adjacencyList[v2][v1].erase(key);
        allEdges.erase(id);
    };

    void addNode(int id, int posx, int posy) {
        allNodes[id] = Node(id, posx, posy);
        if (id > maxNodeId) maxNodeId = id;
    };

    void removeNode(int node) {
        std::vector<Edge> edges = getEdges(node);
        for (const auto& edge : edges) {
            allEdges.erase(edge.id);
        };

        std::map<int, std::map<int, Edge>> nodeToKeysToEdges = adjacencyList[node];
        adjacencyList.erase(node);

        for (const auto& pair : nodeToKeysToEdges) {
            adjacencyList[pair.first].erase(node);
        };

        allNodes.erase(node);
    };

    std::vector<Edge> getAllEdges() {
        std::vector<Edge> edges;
        for (const auto& pair : allEdges) {
            edges.push_back(pair.second);
        };
        return edges;
    };

    std::vector<Edge> getEdges(int node) {
        std::vector<Edge> edges;
        for (const auto& pair1 : adjacencyList[node]) {
            for (const auto& pair2 : pair1.second) {
                edges.push_back(pair2.second);
            };
        };
        return edges;
    };

    std::vector<Edge> getEdges(int node1, int node2) {
        std::vector<Edge> edges;
        for (const auto& pair : adjacencyList[node1][node2]) {
            edges.push_back(pair.second);
        };
        return edges;
    };

    Edge getMinCostEdge(int node1, int node2) {
        std::optional<Edge> edge;
        double minCost = MAX_DOUBLE;
        for (const auto& pair : adjacencyList[node1][node2]) {
            if (pair.second.cost < minCost) {
                edge = pair.second;
                minCost = edge->cost;
            };
        };

        if (!edge) {
            throw std::runtime_error("expected to find edge between node " + std::to_string(node1) + " and " + std::to_string(node2));
        }
        return *edge;
    };

    Edge getMinCostEdge(int node1, int node2, std::set<int> exceptEdges) {
        std::optional<Edge> edge;
        double minCost = MAX_DOUBLE;
        for (const auto& pair : adjacencyList[node1][node2]) {
            if (pair.second.cost < minCost) {
                // filter out edges from exceptEdges
                if (exceptEdges.contains(pair.second.id)) continue;

                edge = pair.second;
                minCost = edge->cost;
            };
        };

        if (!edge) {
            throw std::runtime_error("expected to find edge between node " + std::to_string(node1) + " and " + std::to_string(node2));
        }

        return *edge;
    };

    Edge getEdgeById(int edgeId) {
        return allEdges[edgeId];
    }

    std::vector<int> getAllEdgeIds() {
        std::vector<int> edges;
        for (const auto& pair : allEdges) {
            edges.push_back(pair.first);
        };
        return edges;
    };

    std::vector<Node> getAllNodes() {
        std::vector<Node> nodes;
        for (const auto& pair : allNodes) {
            nodes.push_back(pair.second);
        };
        return nodes;
    };

    std::vector<int> getAllNodeIds() {
        std::vector<int> nodes;
        for (const auto& pair : allNodes) {
            nodes.push_back(pair.first);
        };
        return nodes;
    };

    std::tuple<int, int> maxPos() {
        int maxx = 0;
        int maxy = 0;
        for (const auto& node : allNodes) {
            if (node.second.posx > maxx) {
                maxx = node.second.posx;
            };
            if (node.second.posy > maxy) {
                maxy = node.second.posy;
            };
        };
        return { maxx, maxy };
    };

    int mergeNodes(const std::set<int> nodes) {
        std::vector<int> allNodes = getAllNodeIds();
        int newNode = (*std::max_element(allNodes.begin(), allNodes.end())) + 1;
        addNode(newNode, -1, -1);

        std::map<int, Edge> edges;
        for (const auto& node : nodes) {
            std::vector<Edge> edgesOfNode = getEdges(node);
            for (const auto& edge : edgesOfNode) {
                edges[edge.id] = edge;
            }
        };

        for (const auto& node : nodes) {
            removeNode(node);
        };

        for (const auto& pair : edges) {
            if (nodes.contains(pair.second.node2)) {
                addEdge(newNode, newNode, pair.second.cost, pair.second.id);
            } else {
                addEdge(newNode, pair.second.node2, pair.second.cost, pair.second.id);
            };
        };

        return newNode;
    };

    std::tuple<std::vector<std::tuple<int, int, int>>, std::vector<std::tuple<int, int, int, double, int>>> export_() {
        std::vector<std::tuple<int, int, int>> nodes;
        std::vector<std::tuple<int, int, int, double, int>> edges;

        for (const auto& pair : allNodes) {
            nodes.push_back({ pair.second.id, pair.second.posx, pair.second.posy });
        };

        for (const auto& pair : allEdges) {
            edges.push_back({ pair.second.node1, pair.second.node2, pair.second.key, pair.second.cost, pair.second.id });
        };

        return { nodes, edges };
    };
};

class Solver {
private:
    Graph graph;
    Graph searchGraph;
    std::map<int, int> nodeToComponent;
    std::map<int, std::set<int>> components;
    std::map<int, std::map<int, double>> nodeToPredecessor;
    std::map<int, std::map<int, double>> foundPaths;
    std::set<int> multicut;
    std::vector<int> searchHistory;
    std::vector<int> madeCut;
    std::vector<int> madeMerge;
    std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>> fullHistory;
    mutable std::mutex mtx;
    bool trackHistory;
    std::ofstream file;
    std::string filename;
    std::map<int, int> componentAge;
    int nextComponentId;
    std::chrono::milliseconds elapsed;

public:
    Solver() {
        trackHistory = false;
        nextComponentId = -1;
    };

    void activateTrackHistory() {
        trackHistory = true;
    };

    float getElapsedTime() {
        return static_cast<float>(elapsed.count());
    }
    
    double getScore() {
        double score = 0;
        for (const int edgeId : multicut) {
            auto edge = graph.getEdgeById(edgeId);
            score += edge.cost;
        }
        return score;
    }

    void loadGraph(const std::vector<std::tuple<int, int, int>>& nodes, const std::vector<std::tuple<int, int, int, double, int>>& edges) {
        py::print("loading graph");

        for (const auto& node : nodes) {
            graph.addNode(std::get<0>(node), std::get<1>(node), std::get<2>(node));
        };

        for (const auto& edge : edges) {
            graph.addEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge), std::get<3>(edge), std::get<4>(edge));
        };

        buildSearchGraph();
    };

    void loadSearchGraph(const std::vector<std::tuple<int, int, int>>& nodes, const std::vector<std::tuple<int, int, int, double, int>>& edges) {
        py::print("loading search graph");

        for (const auto& node : nodes) {
            searchGraph.addNode(std::get<0>(node), std::get<1>(node), std::get<2>(node));
        }

        for (const auto& edge : edges) {
            searchGraph.addEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge), std::get<3>(edge), std::get<4>(edge));
        }

        py::print("loading search graph finished");
    };

    void buildSearchGraph() {
        // this method assumes that the id of nodes is labeled by rows; increasing with growing x and continueing with the next y y;

        py::print("building search graph");
        int maxx, maxy;
        std::tie(maxx, maxy) = graph.maxPos();
        int numNodes = (maxx + 1) * (maxy + 1);
        int colCount = maxx + 1;
        int outsideNode = numNodes + 1;
        searchGraph.addNode(outsideNode, -1, -1);
        int x, y;
        Edge edgeRightTop, edgeLeftDown;

        for (int i = 0; i < numNodes; i++) {
            x = i % colCount;
            y = i / colCount;

            if (x != maxx && y != maxy) {
                searchGraph.addNode(i, x, y);
                edgeRightTop = graph.getEdges(i, i + 1)[0];
                edgeLeftDown = graph.getEdges(i, i + colCount)[0];
                if (y == 0) {
                    searchGraph.addEdge(i, outsideNode, edgeRightTop.cost, edgeRightTop.id);
                } else {
                    searchGraph.addEdge(i, i - colCount, edgeRightTop.cost, edgeRightTop.id);
                };
                if (x == 0) {
                    searchGraph.addEdge(i, outsideNode, edgeLeftDown.cost, edgeLeftDown.id);
                } else {
                    searchGraph.addEdge(i, i - 1, edgeLeftDown.cost, edgeLeftDown.id);
                };
            } else {
                if (x != maxx) {
                    edgeRightTop = graph.getEdges(i, i + 1)[0];
                    searchGraph.addEdge(outsideNode, i - colCount, edgeRightTop.cost, edgeRightTop.id);
                };

                if (y != maxy) {
                    edgeLeftDown = graph.getEdges(i, i + colCount)[0];
                    searchGraph.addEdge(outsideNode, i - 1, edgeLeftDown.cost, edgeLeftDown.id);
                };
            };
        };

        py::print("nodes:", searchGraph.getAllNodes().size());
        py::print("edges:", searchGraph.getAllEdges().size());
    };

    int getNewComponentId() {
        if (nextComponentId == -1) {
            auto it = components.rbegin();
            int greatestKey = (it != components.rend()) ? it->first : 0;
            nextComponentId = greatestKey + 1;
        }
        return nextComponentId++;
    };

    std::tuple<int, double> getLowestCostPredecessor(int node) {
        int minCostPredecessor = node;
        double minCost = 0;

        for (const auto& [predecessor, cost] : nodeToPredecessor[node]) {
            if (node == predecessor) continue;
            if (cost < minCost) {
                minCostPredecessor = predecessor;
                minCost = cost;
            };
        };

        return {minCostPredecessor, minCost};
    };

    std::tuple<int, double> getLowestCostPredecessor(int node, int exceptNode) {
        int minCostPredecessor;
        double minCost = 0;

        for (const auto& [predecessor, cost] : nodeToPredecessor[node]) {
            if (node == predecessor) continue;
            if (predecessor == exceptNode) continue;
            if (cost < minCost) {
                minCostPredecessor = predecessor;
                minCost = cost;
            };
        };

        return { minCostPredecessor, minCost };
    };

    void handleSelfEdgesOfNode(int node) {
        for (const Edge& edge : searchGraph.getEdges(node)) {
            if (edge.node1 == edge.node2) {
                if (edge.cost < 0) {
                    multicut.insert(edge.id);
                };

                auto it1 = nodeToPredecessor[edge.node1].find(edge.node2);
                if (it1 != nodeToPredecessor[edge.node1].end()) {
                    nodeToPredecessor[edge.node1].erase(edge.node2);
                };

                searchGraph.removeEdge(edge.node1, edge.node2, edge.key, edge.id);
            };
        };
    };

    void handleSelfEdges() {
        for (const Edge& edge : searchGraph.getAllEdges()) {
            if (edge.node1 == edge.node2) {
                if (edge.cost < 0) {
                    multicut.insert(edge.id);
                };

                auto it1 = nodeToPredecessor[edge.node1].find(edge.node2);
                if (it1 != nodeToPredecessor[edge.node1].end()) {
                    nodeToPredecessor[edge.node1].erase(edge.node2);
                };

                searchGraph.removeEdge(edge.node1, edge.node2, edge.key, edge.id);
            };
        };
    };

    void updateComponentCost(int componentId) {
        std::set<int> nodes = components[componentId];
        std::vector<int> startNodes;
        for (const auto& node : nodes) {
            if (nodeToPredecessor[node].size() < 2) {
                startNodes.push_back(node);
            };
        };

        std::queue<std::tuple<int, int>> nodeQueue;
        for (const auto& startNode : startNodes) {

            nodeQueue = std::queue<std::tuple<int, int>>();
            int predecessor = startNode;
            if (nodeToPredecessor[startNode].size() == 0) continue;
            int secondNode = nodeToPredecessor[startNode].begin()->first;
            nodeQueue.push({secondNode, predecessor});
            while (!nodeQueue.empty()) {
                auto [currentNode, predecessor] = nodeQueue.front();
                nodeQueue.pop();

                if (currentNode == predecessor) continue;

                double predecessorCost = std::get<1>(getLowestCostPredecessor(predecessor, currentNode));
                double edgeCost = searchGraph.getMinCostEdge(currentNode, predecessor).cost;

                nodeToPredecessor[currentNode][predecessor] = predecessorCost + edgeCost;

                for (const auto& pair : nodeToPredecessor[currentNode]) {
                    if (pair.first == predecessor) continue;
                    nodeQueue.push({ pair.first, currentNode });
                };
            };
        };
    };

    void updateComponentCostInDirection(int componentId, int node1, int node2) {
        std::queue<std::tuple<int, int>> nodeQueue;
        nodeQueue.push({ node2, node1 });
        double predecessorCost, edgeCost, oldCost, newCost;
        while (!nodeQueue.empty()) {
            auto [currentNode, predecessor] = nodeQueue.front();
            nodeQueue.pop();

            if (currentNode == predecessor) continue;

            predecessorCost = std::get<1>(getLowestCostPredecessor(predecessor, currentNode));
            edgeCost = searchGraph.getMinCostEdge(currentNode, predecessor).cost;

            newCost = predecessorCost + edgeCost;
            oldCost = nodeToPredecessor[currentNode][predecessor];

            if (newCost == oldCost) continue;
            nodeToPredecessor[currentNode][predecessor] = newCost;

            for (const auto& pair : nodeToPredecessor[currentNode]) {
                if (pair.first == predecessor) continue;
                nodeQueue.push({ pair.first, currentNode });
            };
        };
    };

    std::tuple<std::vector<std::tuple<int, Edge>>, double> findPath(int startNode, int endNode) {

        if (startNode == endNode) {
            std::vector<std::tuple<int, Edge>> emptyPath;
            return { emptyPath, 0 };
        };

        std::map<int, Edge> predecessors;
        std::vector<std::tuple<int, double>> nodeQueue;
        auto currentNode = startNode;
        double cost = 0;
        std::set<int> foundEdges;
        std::set<int> foundNodes;
        foundNodes.insert(startNode);

        while (true) {
            for (const auto& pair : nodeToPredecessor[currentNode]) {
                int neighbor = pair.first;

                if (foundNodes.contains(neighbor)) continue;

                Edge minEdge = searchGraph.getMinCostEdge(currentNode, neighbor, foundEdges);

                double totalNeighborCost = cost + minEdge.cost;
                predecessors[neighbor] = minEdge;

                // mark edge as found
                foundEdges.insert(minEdge.id);

                // mark node as found
                foundNodes.insert(neighbor);

                if (neighbor == endNode) {
                    // found path
                    Edge edge = predecessors[endNode];
                    int predecessorNode = edge.node1;
                    std::vector<std::tuple<int, Edge>> path;

                    while (predecessorNode != startNode) {
                        path.emplace_back(predecessorNode, edge);
                        edge = predecessors[predecessorNode];
                        predecessorNode = edge.node1;
                    };
                    path.emplace_back(startNode, edge);
                    std::reverse(path.begin(), path.end());

                    return { path, totalNeighborCost };
                };
                nodeQueue.emplace_back(neighbor, totalNeighborCost);
            };

            if (nodeQueue.empty()) break;
            std::tie(currentNode, cost) = nodeQueue.back();
            nodeQueue.pop_back();
        };

        throw std::runtime_error("findPath expected to find a path, but no path was found.");
    };

    int handleCycle(std::vector<std::tuple<int, Edge>> cycle, bool createComponent = true) {
        // remove cut edges from graph and add them to the multicut
        for (const auto& [node, edge] : cycle) {
            if (debug) py::print(node, edge.cost);
            searchGraph.removeEdge(edge.node1, edge.node2, edge.key, edge.id);
            multicut.insert(edge.id);
            if (trackHistory) madeCut.push_back(edge.id);

        };

        // create necessary variables
        std::set<int> cycleNodesSet;
        std::transform(cycle.begin(), cycle.end(), std::inserter(cycleNodesSet, cycleNodesSet.begin()),
            [](const auto& tuple) { return std::get<0>(tuple); });

        std::set<int> componentIdsSet;
        std::transform(cycleNodesSet.begin(), cycleNodesSet.end(), std::inserter(componentIdsSet, componentIdsSet.begin()),
            [this](const auto& node) { return nodeToComponent[node]; });
        
        std::set<int> allNodesSet;
        for (const auto& componentId : componentIdsSet) {
            std::set_union(allNodesSet.begin(), allNodesSet.end(), components[componentId].begin(), components[componentId].end(), std::inserter(allNodesSet, allNodesSet.begin()));
            // remove component
            components.erase(componentId);
        };

        std::set<int> nonCycleNodesSet;
        std::set_difference(allNodesSet.begin(), allNodesSet.end(),
            cycleNodesSet.begin(), cycleNodesSet.end(), std::inserter(nonCycleNodesSet, nonCycleNodesSet.begin()));

        // remove component
        for (const auto& node : allNodesSet) {
            nodeToComponent.erase(node);
        };

        // merge nodes
        int newNode = searchGraph.mergeNodes(cycleNodesSet);

        for (const auto& cycleNode : cycleNodesSet) {
            for (const auto& pair : nodeToPredecessor[cycleNode]) {
                int neighbour = pair.first;
                if (cycleNodesSet.contains(neighbour)) continue;
                nodeToPredecessor[neighbour].erase(cycleNode);
                if (createComponent) {
                    nodeToPredecessor[neighbour][newNode] = 0;
                    nodeToPredecessor[newNode][neighbour] = std::get<1>(getLowestCostPredecessor(neighbour));
                };
            };
            nodeToPredecessor.erase(cycleNode);
        };

        handleSelfEdgesOfNode(newNode);

        if (createComponent) {
            nonCycleNodesSet.insert(newNode);
            int newComponentId = getNewComponentId();
            components[newComponentId] = nonCycleNodesSet;
            for (const auto& node : nonCycleNodesSet) {
                nodeToComponent[node] = newComponentId;
            };

            for (const auto& pair : nodeToPredecessor[newNode]) {
                updateComponentCostInDirection(newComponentId, newNode, pair.first);
            }
            return newComponentId;
        } else {
            // reset all variables
            // this could be optimized
            nodeToComponent.clear();
            nodeToPredecessor.clear();
            components.clear();
            return -1;
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
                std::set<int> newComponent;
                newComponent.insert(nodeId);
                int newComponentId = getNewComponentId();
                components[newComponentId] = newComponent;
                nodeToComponent[nodeId] = newComponentId;
                componentAge[newComponentId] = 0;

                // add all nodes to component that have negative cost
                std::map<int, int> previousSearchedNodes;
                int previousSearchedNode;
                std::vector<int> unsearchedNodes;
                unsearchedNodes.push_back(nodeId);
                while (!unsearchedNodes.empty()) {
                    if (reset) break;
                    int unsearchedNode = unsearchedNodes.back();

                    unsearchedNodes.pop_back();

                    auto it = previousSearchedNodes.find(unsearchedNode);
                    if (it != previousSearchedNodes.end()) {
                        previousSearchedNode = previousSearchedNodes[unsearchedNode];
                    };

                    for (const Edge& edge : searchGraph.getEdges(unsearchedNode)) {
                        int newNode = edge.node2;

                        if (it != previousSearchedNodes.end() && previousSearchedNode == newNode) continue;

                        if (edge.cost < 0) {
                            // check for cycle
                            if (newComponent.contains(newNode)) {
                                // cycle found
                                auto [path, cost] = findPath(newNode, unsearchedNode);

                                // add last node and edge to make the path to a cycle
                                cost += edge.cost;
                                path.push_back({ unsearchedNode, edge });

                                handleCycle(path, false);

                                reset = true;
                                break;
                            }

                            previousSearchedNodes[newNode] = unsearchedNode;

                            nodeToPredecessor[newNode][unsearchedNode] = 0;
                            nodeToPredecessor[unsearchedNode][newNode] = 0;

                            // add node to component
                            newComponent.insert(newNode);
                            components[newComponentId] = newComponent;
                            unsearchedNodes.push_back(newNode);
                            nodeToComponent[newNode] = newComponentId;

                            // update costs
                            updateComponentCost(newComponentId);
                        };
                    };
                };
            };
        };
    };

    void mergeComponents(int swallower, int swallowee) {
        if (swallower == swallowee) {
            throw std::runtime_error("Merging a component into itself is not possible.");
        };

        std::set_union(components[swallower].begin(), components[swallower].end(), components[swallowee].begin(), components[swallowee].end(), std::inserter(components[swallower], components[swallower].begin()));

        for (const auto& node : components[swallowee]) {
            nodeToComponent[node] = swallower;
        };

        components.erase(swallowee);
    };

    // cost, node, start node, component, age
    typedef std::tuple<double, int, int, int, int> node_tuple;

    bool searchParallel(bool allowCuts = true) {
        // initialize priority queue with all nodes with negative cost
        // save for every node in priority queue: full cost, node, start node
        // save distances for every node
        // save predecessor for every node
        // save start node for every node
        py::print("search parallel");

        if (trackHistory) {
            searchHistory.clear();
            madeCut.clear();
            madeMerge.clear();
        };

        foundPaths.clear();

        std::priority_queue<node_tuple, std::vector<node_tuple>, std::greater<node_tuple>> priorityQueue;
        int maxNodes = (searchGraph.getMaxNodeId() + 1) * 2;

        // totalCost optimal from that node
        std::vector<double> costVec(maxNodes, INT_MAX);

        // distance to startNode
        std::vector<double> dist(maxNodes, INT_MAX);
        // predecessor in search
        std::vector<int> pred(maxNodes, -1);
        // start node from where the node is reached
        std::vector<int> start(maxNodes, -1);

        std::vector<int> fromComponent(maxNodes, -1);

        std::vector<int> nodeAge(maxNodes, -1);

        // for node in nodes with negative cost
        // initialize vectors and priority queue
        for (const auto node : searchGraph.getAllNodeIds()) {
            double cost = std::get<1>(getLowestCostPredecessor(node));
            // initialize all nodes part from a bigger component
            if (nodeToPredecessor[node].size() > 0) {
                int component = nodeToComponent[node];
                priorityQueue.push({ cost, node, node, component, componentAge[component]});
                costVec[node] = cost;
                dist[node] = 0;
                pred[node] = node;
                start[node] = node;
                fromComponent[node] = component;
                nodeAge[node] = componentAge[component];
            };
        };

        bool fullContinue = false;

        while (!priorityQueue.empty()) {
            auto [cost, node, nodeStartNode, nodeComponent, nodeComponentAge] = priorityQueue.top();
            priorityQueue.pop();

            if (nodeComponentAge < componentAge[nodeComponent]) continue;

            double nodeDist = dist[node];

            // check all connected nodes via positive edges
            for (const auto& edge : searchGraph.getEdges(node)) {
                if (edge.cost > 0) {
                    int newNode = edge.node2;
                    int newNodeAge = nodeAge[newNode];
                    double newCost = edge.cost + cost;
                    double newDist = edge.cost + nodeDist;

                    int newNodeStartNode = start[newNode];

                    // check if node is connected with newNode in component (if there are multiple edges between these nodes, this is too resctrictive)
                    if (nodeToPredecessor[node].count(newNode) > 0) continue;

                    // check if node is known and if the cost is not outdated
                    if (newNodeAge >= componentAge[fromComponent[newNode]]) {
                        // found path or cycle
                        int nodeComponent = nodeToComponent[nodeStartNode];
                        int newNodeComponent = nodeToComponent[newNodeStartNode];
                        if (nodeComponent == newNodeComponent) {
                            if (!allowCuts) continue;
                            // found cycle
                            auto it = foundPaths.find(nodeStartNode);
                            if (it != foundPaths.end()) {
                                auto it2 = (it->second).find(newNodeStartNode);
                                if (it2 != (it->second).end()) {
                                    if (it2->second + newDist + dist[newNode] >= 0) continue;
                                }
                            }

                            auto [path, cost] = findPath(nodeStartNode, newNodeStartNode);

                            foundPaths[nodeStartNode][newNodeStartNode] = cost;
                            foundPaths[newNodeStartNode][nodeStartNode] = cost;

                            // ignore cycle if cost is greater or equal to zero
                            if (cost + newDist + dist[newNode] >= 0) continue;

                            // handle cycle
                            int iterNode = node;
                            int iterPredecessor = pred[node];

                            // add path from node to nodeStartNode
                            int i = 0;
                            
                            while (iterNode != iterPredecessor) {
                                Edge edge = searchGraph.getMinCostEdge(iterNode, iterPredecessor);
                                path.insert(path.begin() + i++, { iterNode, edge });
                                componentAge[nodeToComponent[iterNode]]++;
                                iterNode = iterPredecessor;
                                iterPredecessor = pred[iterPredecessor];
                            };
                            componentAge[nodeToComponent[iterNode]]++;

                            // add path from newNode to newNodeStartNode
                            iterNode = newNode;
                            iterPredecessor = pred[newNode];
                            Edge edge = searchGraph.getMinCostEdge(newNode, node);
                            i = 0;
                            while (iterNode != iterPredecessor) {
                                path.insert(path.end() - i++, { iterNode, edge });
                                edge = searchGraph.getMinCostEdge(iterPredecessor, iterNode);
                                componentAge[nodeToComponent[iterNode]]++;
                                iterNode = iterPredecessor;
                                iterPredecessor = pred[iterPredecessor];
                            };
                            componentAge[nodeToComponent[iterNode]]++;
                            path.insert(path.end() - i++, { iterNode, edge });

                            int newComponent = handleCycle(path);
                            
                            componentAge[nodeComponent]++;
                            
                            componentAge[newComponent] = 0;

                            for (const auto node : components[newComponent]) {
                                double cost = std::get<1>(getLowestCostPredecessor(node));
                                priorityQueue.push({ cost, node, node, newComponent, 0 });
                                costVec[node] = cost;
                                dist[node] = 0;
                                pred[node] = node;
                                start[node] = node;
                                fromComponent[node] = newComponent;
                                nodeAge[node] = 0;
                            };

                            // think about what happens after cycle was cut, maybe restart?
                            fullContinue = true;
                            break;
                        } else {
                            // found path between two components

                            nodeToPredecessor[node][newNode] = 0;
                            nodeToPredecessor[newNode][node] = 0;

                            if (trackHistory) madeMerge.push_back(searchGraph.getMinCostEdge(node, newNode).id);

                            // add nodes between node and nodeStartNode
                            int iterNode = node;
                            int iterPredecessor = pred[node];
                            int node1 = node;
                            int node2 = newNode;
                            int node3 = newNode;
                            int node4 = node;
                            int iterNodeComponent;

                            while (iterNode != iterPredecessor) {

                                if (trackHistory) madeMerge.push_back(searchGraph.getMinCostEdge(iterNode, iterPredecessor).id);

                                nodeToPredecessor[iterNode][iterPredecessor] = 0;
                                nodeToPredecessor[iterPredecessor][iterNode] = 0;
                                iterNodeComponent = nodeToComponent[iterNode];
                                if (nodeComponent != iterNodeComponent) {
                                    componentAge[iterNodeComponent]++;
                                    mergeComponents(nodeComponent, iterNodeComponent);
                                };

                                node1 = iterPredecessor;
                                node2 = iterNode;
                                iterNode = iterPredecessor;
                                iterPredecessor = pred[iterPredecessor];
                            };

                            // add nodes between newNode and newNodeStartNode
                            iterNode = newNode;
                            iterPredecessor = pred[newNode];

                            while (iterNode != iterPredecessor) {
                                // py::print(iterNode, iterPredecessor);
                                if (trackHistory) madeMerge.push_back(searchGraph.getMinCostEdge(iterNode, iterPredecessor).id);

                                nodeToPredecessor[iterNode][iterPredecessor] = 0;
                                nodeToPredecessor[iterPredecessor][iterNode] = 0;
                                iterNodeComponent = nodeToComponent[iterNode];
                                if (nodeComponent != iterNodeComponent) {
                                    componentAge[iterNodeComponent]++;
                                    mergeComponents(nodeComponent, iterNodeComponent);
                                };

                                node3 = iterPredecessor;
                                node4 = iterNode;
                                iterNode = iterPredecessor;
                                iterPredecessor = pred[iterPredecessor];
                            };
                            
                            componentAge[newNodeComponent]++;
                            componentAge[nodeComponent]++;
                            mergeComponents(nodeComponent, newNodeComponent);

                            updateComponentCostInDirection(nodeComponent, node1, node2);
                            updateComponentCostInDirection(nodeComponent, node3, node4);


                            // continue by reinitializing all nodes from the new component to the prority queue, existing values in the queue will be ignored when they are reached
                            int age = componentAge[nodeComponent];
                            for (const auto node : components[nodeComponent]) {
                                double cost = std::get<1>(getLowestCostPredecessor(node));
                                priorityQueue.push({ cost, node, node, nodeComponent, age});
                                costVec[node] = cost;
                                dist[node] = 0;
                                pred[node] = node;
                                start[node] = node;
                                fromComponent[node] = nodeComponent;
                                nodeAge[node] = age;
                            };

                            // continue with next node from priority queue
                            fullContinue = true;
                            break;
                        };
                    };

                    // continue when merge happened
                    if (fullContinue) {
                        if (trackHistory) {
                            setHistory();
                            searchHistory.clear();
                            madeCut.clear();
                            madeMerge.clear();
                        }

                        fullContinue = false;
                        fullContinue = false;
                        continue;
                    }

                    // don't add if cost is not good enough
                    if (newCost >= 0) continue;

                    if (trackHistory) {
                        searchHistory.push_back(searchGraph.getEdges(node, newNode)[0].id);
                    };
                    
                    priorityQueue.push({ newCost, newNode, nodeStartNode, nodeComponent, nodeComponentAge });
                    costVec[newNode] = newCost;
                    dist[newNode] = newDist;
                    pred[newNode] = node;
                    start[newNode] = nodeStartNode;
                    fromComponent[newNode] = nodeComponent;
                    nodeAge[newNode] = nodeComponentAge;
                };
            };
        };

        return false;
    };

    bool searchFrom(int startNode) {
        int startComponentId = nodeToComponent[startNode];

        // reset variables for tracking search history
        if (trackHistory) {
            searchHistory.clear();
            searchHistory.push_back(startNode);
            madeCut.clear();
            madeMerge.clear();
        };

        std::map<int, std::tuple<int, double>> foundNodes;
        double tmpCost = 0;
        foundNodes[startNode] = { startNode, tmpCost };
        std::set<std::tuple<int, int, double>, TupleComparator> nodeQueue;
        int iterNode;
        int iterPredecessor;

        for (const auto& pair : searchGraph.getEdges(startNode)) {
            int neighbor = pair.node2;
            // difference to python implementation
            // check if neigbor is in node to predecessor
            auto it = nodeToPredecessor[startNode].find(neighbor);
            if (it == nodeToPredecessor[startNode].end()) {
                double neighborCost = searchGraph.getMinCostEdge(startNode, neighbor).cost;
                nodeQueue.emplace(neighbor, startNode, neighborCost);
            };
        };

        while (!nodeQueue.empty()) {
            auto firstElement = *nodeQueue.begin();
            auto [node, predecessor, cost] = firstElement;
            nodeQueue.erase(firstElement);
            if (trackHistory) {
                searchHistory.push_back(searchGraph.getEdges(node, predecessor)[0].id);
            };

            foundNodes[node] = { predecessor, cost };

            if (nodeToPredecessor[node].size() > 0 && node != startNode) {
                int componentId = nodeToComponent[node];
                if (componentId == startComponentId) {
                    // found cycle
                    // find path through component and add to cycle
                    auto [path, pathCost] = findPath(startNode, node);

                    if (pathCost + cost >= 0) continue;

                    // find newly found path through positive edges and add to cycle
                    iterNode = node;
                    iterPredecessor = predecessor;
                    while (iterNode != startNode) {
                        Edge edge = searchGraph.getMinCostEdge(iterNode, iterPredecessor);
                        path.emplace_back(iterNode, edge);
                        iterNode = iterPredecessor;
                        iterPredecessor = std::get<0>(foundNodes[iterPredecessor]);
                    };
                    handleCycle(path);
                    return true;
                };

                // merge components and all nodes inbetween

                // update node_to_predecessor to prepare for merging
                iterNode = node;
                iterPredecessor = predecessor;
                while (iterNode != startNode) {

                    if (trackHistory) madeMerge.push_back(searchGraph.getMinCostEdge(node, predecessor).id);

                    nodeToPredecessor[iterNode][iterPredecessor] = 0;
                    nodeToPredecessor[iterPredecessor][iterNode] = 0;
                    if (startComponentId != nodeToComponent[iterNode]) {
                        mergeComponents(startComponentId, nodeToComponent[iterNode]);
                    };

                    iterNode = iterPredecessor;
                    iterPredecessor = std::get<0>(foundNodes[iterPredecessor]);
                };
                updateComponentCost(startComponentId);
                return true;
            };

            // add next nodes to node queue and calculate cost
            for (const auto& edge : searchGraph.getEdges(node)) {
                int neighbor = edge.node2;
                if (neighbor != predecessor && neighbor != node) {
                    double neighborCost = cost + searchGraph.getMinCostEdge(neighbor, node).cost;
                    if (foundNodes.contains(neighbor)) {
                        if (std::get<1>(foundNodes[neighbor]) > neighborCost) {
                            foundNodes[neighbor] = { node, neighborCost };
                        } else {
                            continue;
                        };
                    };
                    nodeQueue.emplace(neighbor, node, neighborCost);
                };
            };
        };

        return false;
    };

    std::tuple<std::vector<std::tuple<int, int, int>>, std::vector<std::tuple<int, int, int, double, int>>, std::map<int, std::set<int>>, std::map<int, std::map<int, double>>> getState() {
        auto [nodes, edges] = searchGraph.export_();

        return { nodes, edges, components, nodeToPredecessor };
    };

    void setHistory() {
        if (!file.is_open()) file.open(filename, std::ios::app);

        for (size_t i = 0; i < searchHistory.size(); ++i) {
            file << searchHistory[i];
            if (i != searchHistory.size() - 1) {
                file << ",";
            }
        }
        file << ";";

        for (size_t i = 0; i < madeCut.size(); ++i) {
            file << madeCut[i];
            if (i != madeCut.size() - 1) {
                file << ",";
            }
        }
        file << ";";

        for (size_t i = 0; i < madeMerge.size(); ++i) {
            file << madeMerge[i];
            if (i != madeMerge.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>> getHistory() {
        std::lock_guard<std::mutex> lock(mtx);
        return fullHistory;
    }

    void search() {
        std::set<int> ignoreNodes;
        int maxIterations = 1000;
        int i = 0;
        while (ignoreNodes.size() < searchGraph.getAllNodeIds().size() and i < maxIterations) {
            double minCost = MAX_DOUBLE;
            int minNode;
            for (const auto& node : searchGraph.getAllNodeIds()) {
                if (ignoreNodes.contains(node)) continue;

                double cost = std::get<1>(getLowestCostPredecessor(node));
                if (cost < minCost) {
                    minCost = cost;
                    minNode = node;
                };
            };

            handleSelfEdges();
            if (!searchFrom(minNode)) {
                ignoreNodes.insert(minNode);
            } else {
                ignoreNodes.clear();
            };
            if (trackHistory) setHistory();
        };
    };

    std::set<int> parallelSearchSolve() {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);

        if (trackHistory) {
            std::stringstream ss;
            ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H-%M-%S");
            filename = ss.str() + ".txt";
            file.open(filename);
            py::print("created file", filename);
        }

        auto start = std::chrono::high_resolution_clock::now();

        py::print("starting initial setup");
        initialSetup();
        py::print("finished initial setup, starting search");

        searchParallel(false);

        if (trackHistory) {
            searchHistory.clear();
            madeCut.clear();
            madeMerge.clear();
            setHistory();
        };
        
        py::print("making cuts");

        searchParallel(true);
        
        auto end = std::chrono::high_resolution_clock::now();

        py::print("finished search");

        file.close();

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        return multicut;
    };
};

Solver getSolver() {
    return Solver();
};


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
        .def("get_state", &Solver::getState)
        .def("get_history", &Solver::getHistory)
        .def("get_score", &Solver::getScore)
        .def("get_elapsed_time", &Solver::getElapsedTime)
        .def("activate_track_history", &Solver::activateTrackHistory)
        .def("parallel_search_solve", &Solver::parallelSearchSolve);

    m.def("get_solver", &getSolver, R"pbdoc(
        Creates a Solver which handles solving the multicut problem.
    )pbdoc");

}
