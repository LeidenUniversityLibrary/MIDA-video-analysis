"""
Python test script for the ALICE user guide.

Multi-processing example
"""

import numpy as np
import os
import socket
from time import time
import multiprocessing as mp

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

    # just for demonstration
    # do not do this here in a production run
    print("(PID {0}) Run {1}: Median of simulation: {2} ".format(pid, run, arr_median))

    return arr_median

if __name__ == "__main__":

    # get starting time of script
    start_time = time()

    print("Python MP test started on {}".format(socket.gethostname()))

    # how many simulation runs:
    n_runs = 100
    size = 10000000

    print("Running {0} simulations of size {1}".format(n_runs, size))

    # Important: only way to get correct core count
    n_cores = os.environ['SLURM_JOB_CPUS_PER_NODE']
    print("The number of cores available from SLURM: {}".format(n_cores))

    # go through the simulations in parallel
    pool = mp.Pool(processes=int(n_cores))
    # use starmap because mysim has multiple inputs
    res = pool.starmap(mysim, [(i,size) for i in range(n_runs)])
    pool.close()
    pool.join()

    print("Python MP test finished (running time: {0:.1f}s)".format(time() - start_time))
