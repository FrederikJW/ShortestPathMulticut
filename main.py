import signal
import sys

from manager import Manager

manager = None


def main():
    global manager
    manager = Manager()
    manager.run_external_spm_solver()


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
