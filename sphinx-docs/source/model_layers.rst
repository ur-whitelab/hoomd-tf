.. _model_layers:

Model Layers
==============

These are standard layers useful for creating molecular neural networks.

RBFExpansion
-------------

A radial basis expansion. TODO

WCARepulsion
--------------

A repulsive layer


.. _eds_biasing:

Biasing with EDS
----------------

To apply `Experiment Directed
Simulation <https://www.tandfonline.com/doi/full/10.1080/08927022.2019.1608988>`__
biasing to a system, use an EDS Layer (:py:class:`utils.EDSLayer`):

.. code:: python

    class EDSModel(htf.SimModel):
        def setup(self):
            self.cv_avg = tf.keras.metrics.Mean()
            self.eds_bias = htf.EDSLayer(4., 5, 1/5)

        def compute(self, nlist, positions, box, sample_weight):
            # get distance from center
            rvec = htf.wrap_vector(positions[0, :3], box)
            # compute CV
            cv = tf.norm(tensor=rvec)
            self.cv_avg.update_state(cv, sample_weight=sample_weight)
            alpha = self.eds_bias(cv)
            # eds energy
            energy = cv * alpha
            forces = htf.compute_positions_forces(positions, energy)
            return forces, alpha

Here,
:py:class:`utils.EDSLayer.update_state`
returns the lagrange multiplier/eds coupling that
is used to bias the simulation.
