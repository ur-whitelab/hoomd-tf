# Hoomd-TF

Markdown: [![status](https://joss.theoj.org/papers/5d1323eadec82aabe86c65a403ff8f90/status.svg)](https://joss.theoj.org/papers/5d1323eadec82aabe86c65a403ff8f90)

This plugin enables the use of TensorFlow in a [Hoomd-blue](http://glotzerlab.engin.umich.edu/hoomd-blue/) simulation to compute quantities like forces and collective variables and do learning while running a simulation. You may also use it without hoomd-blue to process trajectories via [MDAnalysis](https://www.mdanalysis.org/). Please see [here for documentation](https://hoomd-tf.readthedocs.io/en/latest) for install and usage instructions.

HOOMD-TF can be used for a variety of tasks such as online force-matching, online machine learning in HOOMD-blue simulations, and arbitrary collective variable calculations using TensorFlow tensor operations. Because both HOOMD-blue and TensorFlow are GPU-accelerated, HOOMD-TF was designed with speed in mind, and minimizes latency with a GPU-GPU communication scheme. Of particular interest, HOOMD-TF allows for online machine learning with early termination, rather than the more tradditional batch learning approach for MD+ML.

HOOMD-TF includes several utility functions as convenient built-ins, such as:
* RDF calculation
* EDS Biasing (See [this paper](https://www.tandfonline.com/doi/full/10.1080/08927022.2019.1608988))
* Coarse-Grained simulation force matching

In addition to all these, the TensorFlow interface of HOOMD-TF makes implementing arbitrary ML models as easy as it is in TensorFlow, by exposing the HOOMD-blue neighbor list and particle positions to TensorFlow. This enables GPU-accelerated tensor calculations, meaning arbitrary collective variables can be treated in the TensorFlow model framework, as long as they can be expressed as tensor operations on particle positions or neighbor lists.

# Tutorials

See [example notebooks here](https://nbviewer.jupyter.org/github/ur-whitelab/hoomd-tf/tree/master/examples/) to learn about what Hoomd-TF can do.


# Prerequisites

The following packages are required to compile:

* `tensorflow < 2.0`
* `hoomd-blue >= 2.5.2`
* `numpy`
* `tbb-devel (only for hoomd-blue 2.8 and above)`

tbb-devel is required for hoomd-blue 2.8 or above when using the
hoomd-blue conda release. It is not automatically installed when
installing hoomd-blue, so use `conda install -c conda-forge
tbb-devel` to install. The Tensorflow version should be any
Tensorflow 1 release. The higher versions, like 1.14, 1.15, will give
lots of warnings about migrating code to Tensorflow 2.0. It is
recommended you install via pip:

    pip install tensorflow-gpu==1.15.0

# Citation

Please use the following citation:

> HOOMD-TF: GPU-Accelerated, Online Machine Learning in the Hoomd-blue Molecular Dynamics Engine. R Barrett, M Chakraborty, DB Amirkulova,
> HA Gandhi, G Wellawatte, and AD White (2020) *Journal of Open Source Software* doi: 10.21105/joss.02367

&copy; University of Rochester
