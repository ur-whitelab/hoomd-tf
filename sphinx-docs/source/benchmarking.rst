.. _benchmarking:

Benchmarking
------------

The ``benchmark.py`` script in the ``htf/benchmarking`` directory is a convenience script for checking benchmark results. Example
use syntax is shown below. This will run a benchmark trial with 10000 Lennard-Jones particles with HOOMD-blue in GPU configuration,
and save the results as a .txt file in the current working directory. The first argument should be an integer for the number of particles
to simulate. The second should be either "gpu" or "cpu" to indicate the execution mode, and the third indicates where to save the results.
Note that large systems may take some time, as the HOOMD benchmarking utility runs five repeats of a 50,000 timestep simulation with 6,000 steps
of warmup each time. In order to run GPU profiling, make sure you have compiled for GPU (see :ref:`compiling`).

.. code:: bash

    python htf/benchmarking/benchmark.py 10000 gpu

