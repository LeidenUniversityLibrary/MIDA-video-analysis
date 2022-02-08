"""
Python test script for the ALICE user guide.

Serial example
"""

import numpy as np
import os
import socket
from time import time

def mysim(run, size=1000000):
    """
    Function to calculate the median of a random generated array
    """
    # get pid
    pid = os.getpid()

    # initialize
    rng = np.random.default_rng(seed=run)

    # create random array
    rnd_array = rng.random(size)

    # get median
    arr_median = np.median(rnd_array)

    print("(PID {}) Run {}: Median of simulation: {} ".format(pid, run, arr_median))

    return arr_median

if __name__ == "__main__":

    # get starting time of script
    start_time = time()

    print("Python test started on {}".format(socket.gethostname()))

    # how many simulation runs:
    n_runs = 100
    size = 10000000

    print("Running {0} simulations of size {1}".format(n_runs, size))

    # go through the simulations
    for i in range(n_runs):
        # run the simulation
        run_result = mysim(i, size=size)

    print("Python test finished (running time: {0:.1f}s)".format(time() - start_time))
