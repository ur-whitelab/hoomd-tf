# Hoomd-TF

This plugin enables the use of TensorFlow to compute forces in a Hoomd-blue simulation. You can also compute other quantities, like collective variables, and do machine learning.

# Quickstart Tutorial

To compute a `1 / r` pairwise potential with Hoomd-TF:

```python
import hoomd.htf as htf
import tensorflow as tf

########### Graph Building Code ###########
graph = htf.graph_builder(64) # max neighbors = 64
pair_energy = graph.nlist_rinv # nlist_rinv is neighbor 1 / r
particle_energy = tf.reduce_sum(pair_energy, axis=1) # sum over neighbors
forces = graph.compute_forces(energy) # compute forces
graph.save('my_model', forces)

########### Hoomd-Sim Code ################
hoomd.context.initialize()
# this will start TensorFlow, so it goes
# in a with statement for clean exit
with htf.tfcompute('my_model') as tfcompute:
    # create a square lattice
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                        n=[3,3])
    nlist = hoomd.md.nlist.cell()
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.nve(group=hoomd.group.all())
    tfcompute.attach(nlist, r_cut=3.0)
    hoomd.run(1e3)
```

This creates a computation graph whose energy function is `2 / r` and also computes forces and virial for the simulation. The `2` is because the neighborlists in Hoomd-TF are *full* neighborlists (double counted). The Hoomd-blue code starts a simulation of a 9 particle square lattice and simulates it for 1000 timesteps under the potential defined in our Hoomd-TF model. The general process of using Hoomd-TF is to build a TensorFlow computation graph, load the graph, and then attach the graph. See below for more detailed information about Hoomd-TF.

# Compiling

The following packages are required to compile:
```
tensorflow < 2.0
hoomd-blue >= 2.5.0
numpy
tbb-devel
```

tbb-devel is only required if using the "simple" method below. The tensorflow
versions should be any Tensorflow 1 release. The higher versions, like 1.14, 1.15,
will give lots of warnings about migrating code to Tensorflow 2.0.

## Simple Compiling

This method assumes you already have installed hoomd-blue. You could do that,
for example, via `conda install -c conda-forge hoomd`. Here are the complete steps
including that

```bash
pip install tensorflow==1.14.0
conda install -c conda-forge hoomd tbb-devel
git clone https://github.com/ur-whitelab/hoomd-tf
cd hoomd-tf && mkdir build && cd build
cmake ..
make install
```

That's it! You can perform this in a conda environment if desired as well.

## In-Depth Installation Guide

See the documentation for a more complete installation guide and description of the code.

&copy; Andrew White at University of Rochester
