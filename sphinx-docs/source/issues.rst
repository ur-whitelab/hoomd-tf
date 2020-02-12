.. _issues:

Known Issues
============

The following is a list of known issues which have no solution. To
report issues in general, please use the `issue tracker
<https://github.com/ur-whitelab/hoomd-tf/issues>`__.

.. _checkpoint_number:

Using Positions
---------------

The maximum number of checkpoints you can save is limited
to 1 million. Edit the source in `tfmanager.py` if this
is too low for your needs

.. _positions_issues:

Using Positions
---------------

Hoomd re-orders positions to improve performance. If you are using CG
mappings that rely on ordering of positions, be sure to disable this:

.. code:: python

    c = hoomd.context.initialize()
    c.sorter.disable()

.. _exploding_gradients:

Exploding Gradients
-------------------

There is a bug in norms
(https://github.com/tensorflow/tensorflow/issues/12071) that somtimes
prevents optimizers to work well with TensorFlow norms. Note that this
is only necessary if you're summing up gradients, like what is commonly
done in computing gradients in optimizers. This isn't usually an issue
for just computing forces. There are three ways to deal with this:

.. _small_training_rates_issue:

Small Training Rates
~~~~~~~~~~~~~~~~~~~~

When Training something like a Lennard-Jones potential or other ``1/r``
potential, high gradients are possible. You can prevent expoding
gradients by using small learning rates and ensuring variables are
initialized so that energies are finite.

.. _safe_norm_issue:

Safe Norm
~~~~~~~~~

There is a workaround (``graph_builder.safe_norm``) in Hoomd-TF. There
is almost no performance penalty, so it is fine to replace ``tf.norm``
with ``graph_builder.safe_norm`` throughout. This method adds a small
amount to all the norms though, so if you rely on some norms being zero
it will not work well.

.. _clipping_gradients_issue:

Clipping Gradients
~~~~~~~~~~~~~~~~~~

Another approach is to clip gradients instead of using safe\_norm:

.. code:: python

    optimizer = tf.train.AdamOptimizer(1e-4)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

.. _neighbor_lists_issue:

Neighbor Lists
~~~~~~~~~~~~~~

Using a max-size neighbor list is non-ideal, especially in CG
simulations where density is non-uniform.
