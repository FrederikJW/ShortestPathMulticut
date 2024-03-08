import signal
import sys

from manager import Manager

manager = None


def main():
    global manager
    manager = Manager()
    # manager.test_visualizer()
    # manager.run_parallel_spm_solver()
    # manager.run_external_spm_solver()
    # manager.run_parallel_edge_contraction_solver()
    # manager.run_andres_edge_contraction_solver_from_file()
    # manager.full_edge_contraction_test()
    # manager.run_maximum_spanning_tree_continued_solver()
    # manager.run_maximum_spanning_tree_solver()
    # manager.run_maximum_matching_solver()
    # manager.run_full_edge_contraction_benchmark_on_snemi()
    # manager.run_spanning_tree_edge_contraction_solver()
    # manager.run_edge_contraction_solver()
    # manager.multithreading_test()
    manager.make_box_plot()


def on_terminate(signum, frame):
    print("Received termination signal. Cleaning up...")

    global manager
    if manager is not None:
        manager.exit()

    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGTERM, on_terminate)
signal.signal(signal.SIGINT, on_terminate)


if __name__ == "__main__":
    main()
