========
HOOMD-TF
========

This plugin enables the use of TensorFlow in a
HOOMD-blue simulation

HOOMD-TF can be used for a variety of tasks such as online force-matching, online machine learning in HOOMD-blue simulations,
and arbitrary collective variable calculations using TensorFlow tensor operations. Because both HOOMD-blue and TensorFlow are GPU-accelerated,
HOOMD-TF was designed with speed in mind, and minimizes latency with a GPU-GPU communication scheme. Of particular interest,
HOOMD-TF allows for online machine learning with early termination, rather than the more tradditional batch learning approach for MD+ML.

HOOMD-TF includes several utility functions as convenient built-ins, such as:

- RDF calculation
- EDS Biasing (See `this paper <https://www.tandfonline.com/doi/full/10.1080/08927022.2019.1608988>`_)
- Coarse-Grained simulation force matching

In addition to all these, the TensorFlow interface of HOOMD-TF makes implementing arbitrary ML models as easy as it is in TensorFlow, by exposing the HOOMD-blue neighbor list and particle positions to TensorFlow. This enables GPU-accelerated tensor calculations, meaning arbitrary collective variables can be treated in the TensorFlow model framework, as long as they can be expressed as tensor operations on particle positions or neighbor lists.

Citation
---------

|status|

.. |status| image:: https://joss.theoj.org/papers/5d1323eadec82aabe86c65a403ff8f90/status.svg
   :target: https://joss.theoj.org/papers/5d1323eadec82aabe86c65a403ff8f90

Please use the following citation:

   HOOMD-TF: GPU-Accelerated, Online Machine Learning in the Hoomd-blue Molecular Dynamics Engine. R Barrett, M Chakraborty, DB Amirkulova,
   HA Gandhi, G Wellawatte, and AD White (2020) *Journal of Open Source Software* doi: 10.21105/joss.02367


Tutorial
--------

See `example notebooks here <https://nbviewer.jupyter.org/github/ur-whitelab/hoomd-tf/tree/master/examples/>`_ to learn about what Hoomd-TF can do.


.. toctree::
   :maxdepth: 3
   :caption: Table of Contents

   quickstart
   compilation
   changelog.rst
   building_the_model
   running
   utilities
   benchmarking
   unit_tests
   code_of_conduct
   issues

.. toctree::
   :maxdepth: 2
   :caption: API

   htf_package

* :ref:`search`
