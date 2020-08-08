.. _running:

Using a Model in a Simulation
=============================

Use your model like this:

.. code:: python

    import hoomd, hoomd.md
    import hoomd.htf as htf

    ...hoomd initialization code...
    model = MyModel(32)
    tfcompute = htf.tfcompute(model)

    nlist = hoomd.md.nlist.cell()
    tfcompute.attach(nlist, r_cut=3)

    ...other hoomd code...

    hoomd.run(...)


where ``MyModel`` is model you created following the steps in :doc:`building_a_model`,
``nlist`` is a hoomd neighbor list object and ``r_cut`` is the
maximum distance to consider particles as being neighbors. ``nlist``
is optional and is not required if your graph doesn't use the ``nlist``
object (you passed ``0`` as the first arg when building your graph).


.. _logging:

Logging
--------

The default logging level of TensorFlow is relatively noisy. You can reduce
the amount of logged statements via

.. code:: python

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

.. _batching:

Batching
--------

If you do not use molecule batching when building your model (i.e., your model isn't a sub class of :py:class: `.MolSimModel`, you can
optionally split your batches to be smaller than the entire system. This
is set via the ``batch_size`` integer argument to :py:meth:`.tfcompute.attach`.
This can help for high-memory simulations where you cannot spare the GPU memory to
have each tensor be the size of your system.

.. _training:

Training
--------

Training can be done while running your simulation where the labels
are the HOOMD-blue forces. To do this, you must first compile your model
as described in Keras documentation. For example,

.. code:: python

    model.compile('Adam', 'mean_squared_error')

will compile your model to use mean squared error on per-particle
forces (note that the forces tensor contains energy in the last column) as the loss
and the Adam optimizer. To train while running, just add the ``train = True`` arg.

.. code:: python

    tfcompute.attach(train=True)

You can also train less than each step (recommended):

.. code:: python

    tfcompute.attach(train=True, period=100)

.. _model_output:

Model Output
-------------

By default, your model output is not saved except
to send the forces (and possibly virial) to HOOMD-blue.
You can have ``tfcompute`` capture your model output
by adding ``save_output_period=100``. In this case,
output will be saved every 100 steps. Note that
if your model is outputting forces as specified in the
constructor of a ``SimModel``, the forces will not be saved.
Here is a complete example:

.. code:: python

    class MyModel(htf.SimModel):
        def compute(self, nlist, positions, box, sample_weight):
            rinv = htf.nlist_rinv(nlist)
            energy = rinv
            forces = htf.compute_nlist_forces(nlist, energy)
            avg_coord_number = tf.reduce_mean(tf.cast(rinv > 0, tf.float32))
            return forces, energy, avg_coord_number

    model = MyModel)
    tfcompute = htf.tfcompute(model)
    ...
    ...
    tfcompute.attach(nlist, rcut=5.0, save_output_period=100)
    hoomd.run(1000)

    output_energy = tfcompute.outputs[0]
    output_avg_coord_number = tfcompute.outputs[1]