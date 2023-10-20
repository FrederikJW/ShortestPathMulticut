import threading
import time

from graph import GraphFactory

# use a transaction lock to prevent drawing if the graph is changed
drawing_lock = threading.Lock()


class Manager:
    # class which manages everything
    def __init__(self):
        # run visualizer in new thread
        self.visualizer = None
        self.visualization_thread = threading.Thread(target=self.run_visualizer)

    def run(self):
        self.visualization_thread.start()

        while self.visualizer is None:
            pass

        self.visualizer.set_graph(GraphFactory.generate_grid((10, 10)))

        time.sleep(10)

        self.visualizer.set_graph(GraphFactory.generate_grid((20, 5)))

        while self.visualization_thread.is_alive():
            pass

    def run_visualizer(self):
        # pygame should be imported only in the new thread and not in the main thread
        # therefore the visualizer must be imported here

        from visualizer import Visualizer
        self.visualizer = Visualizer(drawing_lock)
        self.visualizer.run()

