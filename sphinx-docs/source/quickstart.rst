Quickstart Tutorial
===================

Here's an example of how you use Hoomd-TF. To compute a ``1 / r``
pairwise potential:

.. code:: python

    import hoomd, hoomd.md
    import hoomd.htf as htf
    import tensorflow as tf

    ########### Graph Building Code ###########
    graph = htf.graph_builder(64) # max neighbors = 64
    pair_energy = graph.nlist_rinv # nlist_rinv is neighbor 1 / r
    particle_energy = tf.reduce_sum(pair_energy, axis=1) # sum over neighbors
    forces = graph.compute_forces(particle_energy) # compute forces
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

This creates a computation graph whose energy function is ``2 / r`` and
also computes forces and virial for the simulation. The ``2`` is because
the neighborlists in Hoomd-TF are *full* neighborlists (double counted).
The Hoomd-blue code starts a simulation of a 9 particle square lattice
and simulates it for 1000 timesteps under the potential defined in our
Hoomd-TF model. The general process of using Hoomd-TF is to build a
TensorFlow computation graph, load the graph, and then attach the graph.
See :ref:`building_the_graph` and :ref:`using_the_graph` for a more detailed
description. Or see a complete set of `Jupyter Notebook tutorials <https://nbviewer.jupyter.org/github/ur-whitelab/hoomd-tf/tree/master/examples/>`_.
