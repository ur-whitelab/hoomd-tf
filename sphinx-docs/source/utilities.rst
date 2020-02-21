.. _utilities:

Utilities
=========

There are a few convenience functions in ``hoomd.htf`` and the
``graph_builder`` class for plotting potential energies of pairwise
potentials and constructing CG mappings.

.. _rdf:

RDF
---

To compute an RDF, use the ``graph.compute_rdf(...)`` method:

.. code:: python

    # set-up graph to compute energy
    ...
    rdf = graph.compute_rdf([1,10], 'rdf', nbins=200)
    graph.running_mean(rdf, 'avg-rdf')
    # run the simulation
    ...
    variables  = htf.load_variables(model_dir, ['avg-rdf'])
    print(variables)

.. _pairwise_potentials:
    
Pairwise Potential and Forces
-----------------------------

To compute a pairwise potential, use the
``graph.compute_pairwise_potential(...)`` method:

.. code:: python

    ...
    r = numpy.arange(1, 10, 1)
    potential, forces = htf.compute_pairwise_potential('/path/to/model', r, potential_tensor)
    ...

.. _eds_biasing:
    
Biasing with EDS
----------------

To apply `Experiment Directed
Simulation <https://www.tandfonline.com/doi/full/10.1080/08927022.2019.1608988>`__
biasing to a system:

.. code:: python

    eds_alpha = htf.eds_bias(cv, set_point=3.0, period=100)
    eds_energy = eds_alpha * cv
    eds_forces = graph.compute_forces(eds_energy)
    graph.save('eds-graph', eds_forces)

where
``htf.eds_bias(cv, set_point, period, learning_rate, cv_scale, name)``
is the function that computes your lagrange multiplier/eds coupling that
you use to bias your simulation. It may be useful to also take the
average of ``eds_alpha`` so that you can use it in a subsequent
simulation:

.. code:: python

    avg_alpha = graph.running_mean(eds_alpha, name='avg-eds-alpha')
    .....
    # after simulation
    vars = htf.load_variables('model/directory', ['avg-eds-alpha'])
    print(vars['avg-eds-alpha'])

.. _coarse_graining:

Coarse-Graining
---------------

Find Molecules
~~~~~~~~~~~~~~

To go from atom index to particle index, use the
``hoomd.htf.find_molecules(...)`` method:

.. code:: python

    # The method takes in a hoomd system as an argument.
    ...
    molecule_mapping_index = hoomd.htf.find_molecules(system)
    ...

Sparse Mapping
~~~~~~~~~~~~~~

The ``sparse_mapping(...)`` method creates the necessary indices and
values for defining a sparse tensor in tensorflow that is a
mass-weighted MxN mapping operator where M is the number of
coarse-grained particles and N is the number of atoms in the system. In
the example,\ ``mapping_per_molecule`` is a list of k x n matrices where
k is the number of coarse-grained sites for each molecule and n is the
number of atoms in the corresponding molecule. There should be one
matrix per molecule. Since the example is for a 1 bead mapping per
molecule the shape is 1 x n. The ordering of the atoms should follow the
output from the find\_molecules method. The variable
``molecule_mapping_index`` is the output from the
``find_molecules(...)`` method.

.. code:: python

    #The example is shown for 1 coarse-grained site per molecule.
    ...
    molecule_mapping_matrix = numpy.ones([1, len(molecule_mapping_index[0])], dtype=np.int)
    mapping_per_molecule = [molecule_mapping_matrix for _ in molecule_mapping_index]
    cg_mapping = htf.sparse_mapping(mapping_per_molecule, \
                        molecule_mapping_index, system = system)
    ...

Center of Mass
~~~~~~~~~~~~~~

The ``center_of_mass(...)`` method maps the given positions according to
the specified mapping operator to coarse-grain site positions
considering periodic boundary condition. The coarse grain site position
is placed at the center of mass of its constituent atoms.

.. code:: python


    ...
    mapped_position = htf.center_of_mass(graph.positions[:,:3], cg_mapping, system)
    #cg_mapping is the output from the sparse_matrix(...) method and indicates how each molecule is mapped.
    ...

Compute Mapped Neighbor List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``compute_nlist(...)`` method returns the neighbor list for the
mapped coarse-grained particles. In the example, ``mapped_position`` is
the mapped particle positions obeying the periodic boundary condition as
returned by the ``center_of_mass(...) method``, ``rcut`` is the cut-off
radius and ``NN`` is the number of nearest neighbors to be considered
for the coarse-grained system.

.. code:: python

    ...
    mapped_nlist= htf.compute_nlist(mapped_position, rcut, NN, system)
    ...

.. _tensorboard:

Tensorboard
-----------

You can visualize your models with tensorboard. First, add
``write_tensorboard=True`` the TensorFlow plugin constructor. This will
add a new directory called ``tensorboard`` to your model directory.

After running, you can launch tensorboard like so:

.. code:: bash

    tensorboard --logdir=/path/to/model/tensorboard

and then visit ``http://localhost:6006`` to view the graph.

Saving Scalars in Tensorboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you would like to save a scalar over time, like total energy or
training loss, you can use the Tensorboard functionality. Add scalars to
the Tensorboard summary during the build step:

.. code:: python

    tf.summary.scalar('total-energy', tf.reduce_sum(particle_energy))

and then add the ``write_tensorboard=True`` flag during the
``tfcompute`` initialize. The period of tensorboard writes is controlled
by the ``saving_period`` flag to the ``tfcompute.attach`` command. View
the Tensorboard section below to see how to view the resulting scalars.

Viewing when TF is running on remote server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are running on a server, before launching tensorboard use this
ssh command to login:

.. code:: bash

    ssh -L 6006:[remote ip or hostname]:6006 username@remote

and then you can view after launching on the server via your local web
browser.

Viewing when TF is running in container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are running docker, you can make this port available a few
different ways. The first is to get the IP address of your docker
container (google how to do this if not default), which is typically
``172.0.0.1``, and then visit ``http://172.0.0.1:6006`` or equivalent if
you have a different IP address for your container.

The second option is to use port forwarding. You can add a port forward
flag, ``-p 6006:6006``, when running the container which will forward
traffic from your container's 6006 port to the host's 6006 port. Again,
then you can visit ``http://localhost:6006`` (linux) or
``http://127.0.0.1:6006`` (windows).

The last method, which usually works when all others fail, is to have
all the container's traffic be on the host. You can do this by adding
the flag ``--net=host`` to the run command of the container. Then you
can visit ``http://localhost:6006``.
    
.. _interactive_mode:

Interactive Mode
----------------

Experimental, but you can trace your graph in realtime in a simulation.
Add both the ``write_tensorboard=True`` to the constructor and the
``_debug_mode=True`` flag to ``attach`` command. You then open another
shell and connect by following the `online instructions for interactive debugging from Tensorboard <https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/debugger#the-debugger-dashboard>`__.
