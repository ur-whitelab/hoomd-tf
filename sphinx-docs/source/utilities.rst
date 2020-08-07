.. _utilities:

Utilities
=============

There are a few convenience functions for plotting potential energies of pairwise
potentials and constructing CG mappings.

.. _rdf:

RDF
---

To compute an RDF, use :py:func:`.compute_rdf`:

.. code:: python

    class LJRDF(htf.SimModel):
        def setup(self):
            self.avg_rdf = tf.keras.metrics.MeanTensor()

        def compute(self, nlist, positions, box, sample_weight):
            # get r
            r = tf.norm(tensor=nlist[:, :, :3], axis=2)
            # compute 1 / r while safely treating r = 0.
            # pairwise energy. Double count -> divide by 2
            inv_r6 = tf.math.divide_no_nan(1., r**6)
            p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
            # rdf from r = 3 to r = 5
            rdf, rs = htf.compute_rdf(nlist, positions, [3, 5])
            # compute running mean
            self.avg_rdf.update_state(rdf, sample_weight=sample_weight)
            forces = htf.compute_nlist_forces(nlist, p_energy)
            return forces

.. _pairwise_potentials:

Pairwise Potential and Forces
-----------------------------

To take your model and compute pairwise outputs,
use :py:func:`.compute_pairwise`, which can
be convenient for computing pairwise energy or forces.

.. code:: python

    model = build_examples.LJModel(4)
    r = np.linspace(0.5, 1.5, 5)
    output = hoomd.htf.compute_pairwise(model, r)

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
``htf.EDSLayer(set_point, period, learning_rate, cv_scale)``
computes the lagrange multiplier/eds coupling that
are used to bias the simulation.

.. _traj_parsing:

Trajectory Parsing
----------------

To process information from a trajectory, use
:py:func:`.iter_from_trajectory`:

.. _coarse_graining:

Coarse-Graining
---------------

Find Molecules
~~~~~~~~~~~~~~

To go from atom index to particle index, use the
:py:func:`.find_molecules`:

.. code:: python

    # The method takes in a hoomd system as an argument.
    ...
    molecule_mapping_index = hoomd.htf.find_molecules(system)
    ...

Sparse Mapping
~~~~~~~~~~~~~~

The :py:func:`.sparse_mapping` method creates the necessary indices and
values for defining a sparse tensor in tensorflow that is a
mass-weighted :math:`M \times N` mapping operator where :math:`M` is the number of
coarse-grained particles and :math:`N` is the number of atoms in the system. In
the following example,\ ``mapping_per_molecule`` is a list of :math:`k \times n` matrices where
:math:`k` is the number of coarse-grained sites for each molecule and :math:`n` is the
number of atoms in the corresponding molecule. There should be one
matrix per molecule. Since the example is for a 1 bead mapping per
molecule the shape is :math:`1 \times n`. The ordering of the atoms should follow the
output from the find\_molecules method. The variable
``molecule_mapping_index`` is the output from
:py:func:`.find_molecules`.

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

:py:func:`.center_of_mass` maps the given positions according to
the specified mapping operator to coarse-grain site positions, while
considering periodic boundary conditions. The coarse grain site position
is placed at the center of mass of its constituent atoms.

.. code:: python


    ...
    mapped_position = htf.center_of_mass(graph.positions[:,:3], cg_mapping, system)
    #cg_mapping is the output from the sparse_matrix(...) method and indicates how each molecule is mapped.
    ...

Compute Mapped Neighbor List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`.compute_nlist` returns the neighbor list for a set of
mapped coarse-grained particles. In the following example, ``mapped_positions`` is
the mapped particle positions obeying the periodic boundary condition, as
returned by  :py:func:`.center_of_mass`, ``rcut`` is the cutoff
radius and ``NN`` is the number of nearest neighbors to be considered
for the coarse-grained system.

.. code:: python

    ...
    mapped_nlist= htf.compute_nlist(mapped_positions, rcut, NN, system)
    ...

.. _tensorboard:

Tensorboard
------------

You can visualize your models with Tensorboard to observe
metrics and other quantities you choose in a web browser. Find out
`more about Tensorboard <https://www.tensorflow.org/tensorboard/get_started>`_.

