.. _building_the_graph:

Building the Graph
==================

To construct a graph, create a :py:class:`htf.graphbuilder.graph_builder` instance:

.. code:: python

    import hoomd.htf as htf
    graph = htf.graph_builder(NN, output_forces)

where ``NN`` is the maximum number of nearest neighbors to consider (can
be 0) and ``output_forces`` indicates if the graph will output forces to
use in the simulation. After building the ``graph``, it will have five
tensors as attributes that can be used when constructing the TensorFlow graph:
``nlist``, ``positions``, ``box``, ``box_size``, and ``forces``:

* ``nlist`` is an ``N`` x ``NN`` x 4 tensor containing the nearest
  neighbors. An entry of all zeros indicates that less than ``NN`` nearest
  neighbors where present for a particular particle. The 4 right-most
  dimensions are ``x,y,z`` and ``w``, which is the particle type. Particle
  type is an integer starting at 0. Note that the ``x,y,z`` values are a
  vector originating at the particle and ending at its neighbor.

* ``positions`` is an ``N`` x 4 tensor of particle positions (x,y,z) and type.
  
* ``forces`` is an ``N`` x 4 tensor that is  *only* available if the graph does
  not output forces (via ``output_forces=False``).

* ``box`` is a 3x3 tensor containing the low box
  coordinate, high box coordinate, and then tilt factors. ``box_size``
  contains just the box length in each dimension.

.. _molecule_batching:

Molecule Batching
-----------------

It may be simpler to have positions or neighbor lists or forces arranged
by molecule. For example, you may want to look at only a particular bond
or subset of atoms in a molecule. To do this, you can call
:py:meth:`htf.graphbuilder.graph_builder.build_mol_rep`, whose argument
``MN`` is the maximum number of atoms
in a molecule. This will create the following new attributes:
``mol_positions``, ``mol_nlist``, and ``mol_forces`` (if your graph has
``output_forces=False``). These new attributes are dimension
``M x MN x ...`` where ``M`` is the number of molecules and ``MN`` is
the atom index within the molecule. If your molecule has fewer than
``MN`` atoms, extra entries will be zeros. You can defnie a molecule to be
whatever you want, and atoms need not be only in one molecule. Here's an
example to compute a water angle, where we assume that the oxygens
are the middle atom:

.. code:: python

    import hoomd.htf as htf
    graph = htf.graph_builder(0)
    graph.build_mol_rep(3)
    # want slice for all molecules (:)
    # want h1 (0), o (1), h2(2)
    # positions are x,y,z,w. We only want x,y z (:3)
    v1 = graph.mol_positions[:, 2, :3] - graph.mol_positions[:, 1, :3]
    v2 = graph.mol_positions[:, 0, :3] - graph.mol_positions[:, 1, :3]
    # compute per-molecule dot product and divide by per molecule norm
    c = tf.einsum('ij,ij->i', v1, v2) / tf.norm(v1, axis=1) / tf.norm(v2 axis=1)
    angles = tf.math.acos(c)

.. _computing_forces:

Computing Forces
----------------

If your graph is outputting forces, you may either compute forces and
pass them to :py:meth:`htf.graphbuilder.graph_builder.save` or have them computed via
automatic differentiation of a potential energy. Call
:py:meth:`htf.graphbuilder.graph_builder.compute_forces` with the argument ``energy``,
which can be either a scalar or a tensor which depends on ``nlist`` and/or ``positions``. A tensor of
forces will be returned as :math:`\sum_i(\frac{-\partial E} {\partial n_i}) - \frac{dE} {dp}`, where the sum is over
the neighbor list. For example, to compute a :math:`1 / r` potential:

.. code:: python

    graph = htf.graph_builder(N - 1)
    #remove w since we don't care about types
    nlist = graph.nlist[:, :, :3]
    #get r
    r = tf.norm(nlist, axis=2)
    #compute 1. / r while safely treating r = 0.
    # halve due to full nlist
    rij_energy = 0.5 * graph.safe_div(1, r)
    #sum over neighbors
    energy = tf.reduce_sum(rij_energy, axis=1)
    forces = graph.compute_forces(energy)

Notice that in the above example that we have used the
:py:meth:`htf.graphbuilder.graph_builder.safe_div` method, which allows
us to safely treat a :math:`1 / 0`, which can arise because ``nlist``
contains 0s for when fewer than ``NN``
nearest neighbors are found.

**Note:** because ``nlist`` is a *full*
neighbor list, you should divide by 2 if your energy is a sum of
pairwise energies.

.. _neighbor_lists:

Neighbor lists
--------------

As mentioned above, :py:class:`htf.graphbuilder.graph_builder` contains a member called
``nlist``, which is an ``N x NN x 4``
neighobr list tensor. You can ask for masked versions of this with
:py:meth:`htf.graphbuilder.graph_builder.masked_nlist`
where ``type_i`` and ``type_j`` are optional integers that specify the type of
the origin (``type_i``) or neighobr (``type_j``). The ``nlist`` argument
allows you to pass in your own neighbor list and ``type_tensor`` allows
you to specify your own list of types, if different than what is given
by hoomd-blue. You can also access ``nlist_rinv`` which gives a
pre-computed ``1 / r`` (dimension ``N x NN``).

.. _virial:

Virial
------

The virial is computed and added to the graph if you use the
:py:meth:`htf.graphbuilder.graph_builder.compute_forces` method
and your energy has a non-zero derivative
with respect to ``nlist``. You may also explicitly pass the virial when
saving, or pass ``None`` to remove the automatically-calculated virial.

.. _finalizing_the_graph:

Finalizing the Graph
--------------------

To finalize and save your graph, you must call
:py:meth:`htf.graphbuilder.graph_builder.save` with the following arguments:

* ``directory``: where to save your TensorFlow model files
* ``force_tensor`` (optional): your computed forces, either as
  computed by your graph or output from :py:meth:`htf.graphbuilder.graph_builder.compute_energy`.
  This should be an ``N x 4`` tensor with the 4th column indicating per-particle potential energy.
* ``virial`` (optional): the virial tensor to save. The virial should be an ``N x 3 x 3`` tensor.
* ``out_nodes`` (optional): If your graph is not outputting forces, then you must provide a tensor or list of
  tensors which will be computed at each timestep. 

.. _saving_data:

Saving Data
-----------

Using variables is the best way to save computed quantities while
running a compute graph. See the :ref:`loading_variables` section for
loading them. You can save a tensor value to a variable using 
:py:meth:`htf.graphbuilder.graph_builder.save_tensor`. Here is an
example of computing the LJ potential and saving the system energy at
each step.

.. code:: python

    # set-up graph
    graph = htf.graph_builder(NN)
    # compute LJ potential
    inv_r6 = graph.nlist_rinv**6
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    energy = tf.reduce_sum(p_energy)
    # save the tensor
    graph.save_tensor(energy, 'lj-energy')
    forces = graph.compute_forces(energy)
    # save the graph
    graph.save(force_tensor=forces, model_directory=directory)

Often you may want a running mean of a variable, for which there is a
built-in, :py:meth:`htf.graphbuilder.graph_builder.running_mean`:

.. code:: python

    # set-up graph to compute energy
    ...
    # we name our variable avg-energy
    graph.running_mean(energy, 'avg-energy')
    # run the simulation
    ...

.. _variables_and_restarts:

Variables and Restarts
----------------------

In TensorFlow, variables are generally trainable parameters. They are
required parts of your graph when doing learning. Each ``save_period``
(set as arg to :py:meth:`htf.tfcompute.tfcompute.attach`),
they are written to your model directory.
Note that when a run is started, the latest values of your
variables are loaded from your model directory. *If you are starting a
new run but you previously ran your model, the old variable values will
be loaded.* To prevent this unexpectedly loading old checkpoints, if you
run :py:meth:`htf.graphbuilder.graph_builder.save`, it will move out all old checkpoints. This
behavior means that if you want to restart, you should not re-run
:py:meth:`htf.graphbuilder.graph_builder.save` in your restart script, *nor* should you pass
``move_previous = False`` as a parameter if you re-run
:py:meth:`htf.graphbuilder.graph_builder.save`.

Variables are also how you save data as seen above. If you are doing
training and also computing other variables, be sure to set your
variables which you do not want to be affected by training optimization
to be ``trainable=False`` when constructing them.

.. _loading_variables:

Loading Variables
-----------------

You may load variables after the simulation using the following syntax:

.. code:: python

    variables  = htf.load_variables(model_dir, ['avg-energy'])

The :py:meth:`htf.utils.load_variables` is general and can be used to load trained,
non-trained, or averaged variables. **It is important to name your custom
variables so they can be loaded using this function.**

.. _period_of_out_nodes:

Period of out nodes
-------------------

You can modify how often tensorflow is called via
:py:meth:`htf.tfcompute.tfcompute.attach`. You can also have more granular control of
operations/tensors passed to ``out_nodes`` by changing the type to a
list whose first element is the tensor and the second argument is the
period at which it is computed. For example:

.. code:: python

    ...graph building code...
    forces = graph.compute_forces(energy)
    avg_force = tf.reduce_mean(forces, axis=-1)
    print_node = tf.Print(energy, [energy], summarize=1000)
    graph.save(force_tensor=forces, model_directory=name, out_nodes=[[print_node, 100], [avg_force, 25]])

This will print the energy every 100 steps and compute the average force
every 25 steps (although it is unused). Note that these two ways of
affecting period both apply. So if the above graph was attached with
``tfcompute.attach(..., period=25)`` then the ``print_node`` will be
run only every 2500 steps.

.. _printing:

Printing
--------

If you would like to print out the values from nodes in your graph, you
can add a print node to the ``out_nodes``. For example:

.. code:: python

    ...graph building code...
    forces = graph.compute_forces(energy)
    print_node = tf.Print(energy, [energy], summarize=1000)
    graph.save(force_tensor=forces, model_directory=name, out_nodes=[print_node])

The ``summarize`` keyword sets the maximum number of numbers to print.
Be wary of printing thousands of numbers per step.

.. _keras_layers:

Optional: Keras Layers for Model Building
-----------------------------------------

Currently HOOMD-TF supports Keras layers in model building. We do not
yet support Keras ``Model.compile()`` or ``Model.fit()``. This example
shows how to set up a neural network model using Keras layers.

.. code:: python

    import tensorflow as tf
    from tensorflow.keras import layers
    import hoomd.htf as htf

    NN = 64
    N_hidden_nodes = 5
    graph = htf.graph_builder(NN, output_forces=False)
    r_inv = graph.nlist_rinv
    input_tensor = tf.reshape(r_inv, shape=(-1,1), name='r_inv')
    #we don't need to explicitly make a keras.Model object, just layers
    input_layer = layers.Input(tensor=input_tensor)
    hidden_layer = layers.Dense(N_hidden_nodes)(input_layer)
    output_layer = layers.Dense(1, input_shape=(N_hidden_nodes,))(hidden_layer)
    #do not call Model.compile, just use the output in the TensorFlow graph
    nn_energies = tf.reshape(output_layer, [-1, NN])
    calculated_energies = tf.reduce_sum(nn_energies, axis=1, name='calculated_energies')
    calculated_forces = graph.compute_forces(calculated_energies)
    #cost and optimizer must also be set through TensorFlow, not Keras
    cost = tf.losses.mean_squared_error(calculated_forces, graph.forces)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    #save using graph.save, not Keras Model.compile
    graph.save(model_directory='/tmp/keras_model/', out_nodes=[ optimizer])

The model can then be loaded and trained as normal. Note that 
``keras.models.Model.fit()`` is not currently supported. You must train
using :py:class:`htf.tfcompute.tfcompute` as explained in the next section.

.. _complete_examples:

Complete Examples
-----------------

The directory `htf/models` contains some example scripts.

.. _lennard_jones_example:

Lennard-Jones with 1 Particle Type
----------------------------------

.. code:: python

    graph = hoomd.htf.graph_builder(NN)
    #use convenience rinv
    r_inv = graph.nlist_rinv
    p_energy = 4.0 / 2.0 * (r_inv**12 - r_inv**6)
    #sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory='/tmp/lj-model')
