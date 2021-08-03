.. _model_layers:

Model Layers
==============

These are standard layers useful for creating molecular neural networks. Only
some are detailed here. See :doc:`layers` for complete list.


WCARepulsion
--------------

The :py:class:`.WCARepulsion` layer can be used to add a trainable
repulsion. Be careful to choose the staring ``sigma`` to be small enough that
there will not be large gradients at the start of training. A regularization
term is added to push ``sigma`` to more positive, otherwise it will just
float away from mattering during training. This can be removed.


.. code:: python

    class WCA(htf.SimModel):
        def setup(self):
            self.wca = htf.WCARepulsion(0.5)

        def compute(self, nlist):
            energy = self.wca(nlist)
            forces = htf.compute_nlist_forces(nlist, energy)
            return forces


.. _eds_biasing:

Biasing with EDS
----------------

To apply `Experiment Directed
Simulation <https://www.tandfonline.com/doi/full/10.1080/08927022.2019.1608988>`__
biasing to a system, use an EDS Layer (:py:class:`.EDSLayer`):

.. code:: python

    class EDSModel(htf.SimModel):
        def setup(self):
            self.cv_avg = tf.keras.metrics.Mean()
            self.eds_bias = htf.EDSLayer(4., 5, 1/5)

        def compute(self, nlist, positions, box):
            # get distance from center
            rvec = htf.wrap_vector(positions[0, :3], box)
            # compute CV
            cv = tf.norm(tensor=rvec)
            self.cv_avg.update_state(cv)
            alpha = self.eds_bias(cv)
            # eds energy
            energy = cv * alpha
            forces = htf.compute_positions_forces(positions, energy)
            return forces, alpha

Here,
:obj:`EDSModel.update_state<tf.keras.metrics.Mean>`
returns the lagrange multiplier/eds coupling that
is used to bias the simulation.
