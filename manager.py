import sys
import threading
import time

import networkx as nx

from graph import GraphFactory
from solver import ShortestPathSolver
from utils import generate_distinct_colors

sys.path.append("C:/Users/FreBe/projects/ShortestPathMulticut/build/Release")
import spm_solver

sys.path.append("C:/Users/FreBe/projects/EdgeContractionMulticut/build/Release")
from edge_contraction_solver import LargestPositiveCost
from edge_contraction_solver import greedyAdditiveEdgeContraction

# use a transaction lock to prevent drawing if the graph is changed
drawing_lock = threading.Lock()


class Manager:
    # class which manages everything
    def __init__(self):
        # run visualizer in new thread
        self.visualizer = None
        self.visualization_thread = threading.Thread(target=self.run_visualizer)

    def multithreading_test(self):
        self.visualization_thread.start()
        while self.visualizer is None:
            time.sleep(1)

        graph = GraphFactory.read_slice_from_snemi3d(50, (200, 200))
        # graph = GraphFactory.generate_grid((30, 30))
        self.visualizer.set_graph(graph)
        solver = spm_solver.Solver()
        solver.activate_track_history()
        solver.load_graph(*(graph.export()))
        nodes, edges, components, node_to_predecessor = solver.get_state()

        graph = GraphFactory.construct_from_values(nodes, edges)
        self.visualizer.set_graph(graph)
        print("start solving")
        solver_thread = threading.Thread(target=solver.solve)
        solver_thread.start()

        self.visualizer.set_history_getter(solver.get_history)

        while self.visualization_thread.is_alive():
            time.sleep(1)

    def test(self):
        self.visualization_thread.start()
        while self.visualizer is None:
            time.sleep(1)

        graph = GraphFactory.read_slice_from_snemi3d(3)
        # graph = GraphFactory.generate_grid((5, 5))
        self.visualizer.set_graph(graph)

        solver = spm_solver.Solver()
        solver.activate_track_history()
        solver.load_graph(*(graph.export()))
        nodes, edges, components, node_to_predecessor = solver.get_state()

        graph = GraphFactory.construct_from_values(nodes, edges)
        self.visualizer.set_graph(graph)
        print("finished")
        while self.visualization_thread.is_alive():
            time.sleep(1)

    def greedy_additive_edge_contraction_test(self):
        multicut, _ = greedyAdditiveEdgeContraction(6, [(0, 1, 5), (0, 3, -20), (1, 2, 5), (1, 4, 5), (2, 5, -20), (3, 4, 5), (4, 5, 5)])
        print("result:")
        print([edge_id for edge_id in multicut])

    def run_official_edge_contraction_solver(self):
        self.visualization_thread.start()

        graph = GraphFactory.generate_grid((50, 50))

        while self.visualizer is None:
            time.sleep(1)

        self.visualizer.set_graph(graph)

        input("Waiting for input")

        multicut, elapsed = greedyAdditiveEdgeContraction(*(graph.standard_export()))

        print("execution Time: ", elapsed, "ms")

        self.visualizer.set_multicut(multicut)

        while self.visualization_thread.is_alive():
            time.sleep(1)

    def run_edge_contraction_solver(self):
        self.visualization_thread.start()

        graph = GraphFactory.generate_grid((50, 50))

        while self.visualizer is None:
            time.sleep(1)

        self.visualizer.set_graph(graph)

        input("Waiting for input")

        solver = LargestPositiveCost()
        solver.load_graph(*(graph.export()))
        multicut, elapsed = solver.solve()

        print("execution Time: ", elapsed, "ms")

        self.visualizer.set_multicut(multicut)

        while self.visualization_thread.is_alive():
            time.sleep(1)

    def run_external_spm_solver(self):
        self.visualization_thread.start()

        graph = GraphFactory.generate_grid((80, 80))

        while self.visualizer is None:
            time.sleep(1)

        self.visualizer.set_graph(graph)

        search_graph = GraphFactory.generate_grid_search_graph(graph)
        visual_graph = search_graph.copy()

        time.sleep(1)
        # input("Waiting for input")

        self.visualizer.set_graph(visual_graph)

        time.sleep(1)
        # input("Waiting for input")

        solver = spm_solver.get_solver()
        solver.load_search_graph(*(search_graph.export()))
        multicut = solver.solve()

        self.visualizer.set_multicut(multicut)

        time.sleep(1)
        # input("Waiting for input")

        self.visualizer.set_graph(graph)

        while self.visualization_thread.is_alive():
            time.sleep(1)

    def run(self):
        self.visualization_thread.start()

        graph = GraphFactory.generate_grid((20, 20))

        while self.visualizer is None:
            time.sleep(1)

        self.visualizer.set_graph(graph)

        # generate graphs to solve and visualize
        search_graph = GraphFactory.generate_grid_search_graph(graph)
        visual_graph = search_graph.copy()
        nx.set_edge_attributes(visual_graph, 0, "cut")

        # solve
        solver = ShortestPathSolver(search_graph)
        multicut = solver.solve()
        components = solver.get_components()

        colors = set(generate_distinct_colors(len(components)))
        component_to_color = dict(enumerate(colors))
        node_to_color = {}
        node_to_value = {}

        color_id = 0
        if len(colors) >= len(components):
            for component_id, component in components.items():
                for node in component:
                    node_to_color[node] = component_to_color[color_id]
                color_id += 1

        for node, mapped_node in solver.get_node_remap().items():
            node_to_value[node] = solver.get_lowest_cost_predecessor(mapped_node)[1]
            node_to_color[node] = node_to_color[mapped_node]

        node_to_value.update({node: solver.get_lowest_cost_predecessor(node)[1] for node in visual_graph.nodes if
                              node not in node_to_value})

        input("Waiting for input")

        self.visualizer.set_graph(visual_graph)

        input("Waiting for input")

        visual_graph.load_values(node_to_value, "cost")
        visual_graph.load_values(node_to_color, "color")

        self.visualizer.set_multicut(multicut)

        input("Waiting for input")

        self.visualizer.set_graph(graph)

        while self.visualization_thread.is_alive():
            time.sleep(1)

    def run_visualizer(self):
        # pygame should be imported only in the new thread and not in the main thread
        # therefore the visualizer must be imported here

        from visualizer import Visualizer
        self.visualizer = Visualizer(drawing_lock)
        self.visualizer.run()

    def exit(self):
        self.visualizer.exit_flag = True
        self.visualization_thread.join()
