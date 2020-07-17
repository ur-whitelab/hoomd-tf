.. _using_the_graph:

Using a Graph in a Simulation
=============================

You may use a saved TensorFlow model via:

.. code:: python

    import hoomd, hoomd.md
    import hoomd.htf as htf

    ...hoomd initialization code...
    with htf.tfcompute(model_dir) as tfcompute:

        nlist = hoomd.md.nlist.cell()
        tfcompute.attach(nlist, r_cut=3)

        ...other hoomd code...

        hoomd.run(...)

where ``model_dir`` is the directory where the TensorFlow model was
saved, ``nlist`` is a hoomd neighbor list object and ``r_cut`` is the
maximum distance for to consider particles as being neighbors. ``nlist``
is optional and is not required if your graph doesn't use the ``nlist``
object (you passed ``NN = 0`` when building your graph).

Logging
--------

The default logging level of Tensorflow is relatively noisy. You can reduce
the amount of logged statements via

.. code:: python
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

Batching
--------

If you used per-molecule positions or nlist in your graph, you can
either rely on hoomd-tf to find your molecules by traversing the bonds
in your system (default) or you can specify what are molecules in your
system. They are passed via ``attach(..., mol_indices=[[..]])``. The
``mol_indices`` are a, possibly ragged, 2D python list where each
element in the list is a list of atom indices for a molecule. For
example, ``[[0,1], [1]]`` means that there are two molecules with the
first containing atoms 0 and 1 and the second containing atom 1. Note
that the molecules can be different size and atoms can exist in multiple
molecules.

If you do not call :py:meth:`graphbuilder.graph_builder.build_mol_rep`
while building your graph, you can
optionally split your batches to be smaller than the entire system. This
is set via the ``batch_size`` integer argument to :py:meth:`tfcompute.tfcompute.attach`.
This can help for high-memory simulations where you cannot spare the GPU memory to
have each tensor be the size of your system.

Bootstrapping Variables
-----------------------

If you have trained variables previously and would like to load them
into the current TensorFlow graph, you can use the ``bootstrap`` and
``bootstrap_map`` arguments. ``bootstrap`` should be a checkpoint file
path or model directory path (latest checkpoint is used) containing
variables which can be loaded into your tfcompute graph. Your model will
be built, then all variables will be initialized, and then your
bootstrap checkpoint will be loaded and no variables will be reloaded
even if there exists a checkpoint in the model directory (to prevent
overwriting your bootstrap variables). ``bootstrap_map`` is an optional
additional argument that will have keys that are variable names in the
``bootstrap`` checkpoint file and values that are names in the tfcompute
graph. This can be used when your variable names do not match up. Here
are two example demonstrating with and without a ``bootstrap_map``:

Here's an example that creates some variables that could be trained
offline without Hoomd. In this example, they just use their initial
values.

.. code:: python

    import tensorflow as tf

    #make some variables
    v = tf.Variable(8.0, name='epsilon')
    s = tf.Variable(2.0, name='sigma')

    #initialize and save them
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, '/tmp/bootstrap/model')

We load them in the hoomd run script:

.. code:: python

    with hoomd.htf.tfcompute(model_dir,
        bootstrap='/tmp/bootstrap/model') as tfcompute:
        ...

Here's how we would load them in the hoomd run script if we want to
change the names of the variables:

.. code:: python

    # here the pretrained variable parameters will replace variables with a different name
    with hoomd.htf.tfcompute(model_dir,
        bootstrap='/tmp/bootstrap/model',
        bootstrap_map={'lj-epsilon':'epsilon', 'lj-sigma':'sigma'}) as tfcompute:
        ...

Bootstrapping Variables from Other Models
-----------------------------------------

Here's an example of bootstrapping where you train with Hoomd-TF and
then load the variables into a different model:

.. code:: python

    # build_models.py
    import tensorflow as tf
    import hoomd.htf as htf

    def make_train_graph(NN, directory):
        # build a model that fits the energy to a linear term
        graph = htf.graph_builder(NN, output_forces=False)
        # get r
        nlist = graph.nlist[:, :, :3]
        r = graph.safe_norm(nlist, axis=2)
        # build energy model
        m = tf.Variable(1.0, name='m')
        b = tf.Variable(0.0, name='b')
        predicted_particle_energy = tf.reduce_sum(m * r + b, axis=1)
        # get energy from hoomd
        particle_energy = graph.forces[:, 3]
        # make them match
        loss = tf.losses.mean_squared_error(particle_energy, predicted_particle_energy)
        optimize = tf.train.AdamOptimizer(1e-3).minimize(loss)
        graph.save(model_directory=directory, out_nodes=[optimize])

    def make_force_graph(NN, directory):
        # this model applies the variables learned in the example above
        # to compute forces
        graph = htf.graph_builder(NN)
        # get r
        nlist = graph.nlist[:, :, :3]
        r = graph.safe_norm(nlist, axis=2)
        # build energy model
        m = tf.Variable(1.0, name='m')
        b = tf.Variable(0.0, name='b')
        predicted_particle_energy = tf.reduce_sum(m * r + b, axis=1)
        forces = graph.compute_forces(predicted_particle_energy)
        graph.save(force_tensor=forces, model_directory=directory)
    make_train_graph(64, 16, '/tmp/training')
    make_force_graph(64, 16, '/tmp/inference')

Here is how we run the training model:

.. code:: python

    #run_train.py
    import hoomd, hoomd.md
    import hoomd.htf as htf


    hoomd.context.initialize()

    with htf.tfcompute('/tmp/training') as tfcompute:
        rcut = 3.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[8,8])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        lj = hoomd.md.pair.lj(rcut, nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(
            group=hoomd.group.all()).randomize_velocities(kT=0.2, seed=42)

        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(100)

Load the variables trained in the training run into the model which
computes forces:

.. code:: python

    #run_inference.py
    import hoomd, hoomd.md
    import hoomd.htf as htf

    hoomd.context.initialize()
    with htf.tfcompute('/tmp/inference',
            bootstrap='/tmp/training') as tfcompute:
        rcut = 3.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[8,8])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        #notice we no longer compute forces with hoomd
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(
            group=hoomd.group.all()).randomize_velocities(kT=0.2, seed=42)

        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(100)

