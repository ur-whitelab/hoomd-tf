Quickstart Tutorial
===================

Here's an example of how you use HOOMD-TF. To compute a ``1 / r``
pairwise potential:

.. code:: python

    import hoomd, hoomd.md
    import hoomd.htf as htf
    import tensorflow as tf

    ########### Model Building Code ###########
    class MyModel(htf.SimModel):
        def compute(self, nlist, positions, box, sample_weight):
            rinv = htf.nlist_rinv(nlist)
            energy = rinv
            forces = htf.compute_nlist_forces(nlist, energy)
            return forces

    ########### Hoomd-Sim Code ################
    hoomd.context.initialize()
    # 32 is maximum number of neighbors
    model = MyModel(32)
    tfcompute = htf.tfcompute(model)
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
the neighbor lists in HOOMD-TF are *full* neighbor lists (double counted).
The HOOMD-blue code starts a simulation of a 9 particle square lattice
and simulates it for 1000 timesteps under the potential defined in our
HOOMD-TF model. The general process of using HOOMD-TF is to build a
model by defining a compute function and then use the model in HOOMD-blue.
See :ref:`building_a_model` and :ref:`running` for a more detailed
description. Or see a complete set of `Jupyter Notebook tutorials <https://nbviewer.jupyter.org/github/ur-whitelab/hoomd-tf/tree/master/examples/>`_.
