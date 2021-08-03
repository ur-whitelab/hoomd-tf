.. _building_a_model:

Building a Model
==================

To modify a simulation, you create a Keras :obj:`tf.keras.Model` that will be executed at
each step (or some multiple of steps) during the simulation. See the :ref:`running`
to see how to train your model instead, though these instructions still apply.

To begin subclass a :py:class:`.SimModel` class:

.. code:: python

    import hoomd.htf as htf
    class MyModel(htf.SimModel):
      def compute(self, nlist, positions, box):
        ...
        return forces, other, important, quantities

    model = MyModel(NN, output_forces=True)

where ``NN`` is the maximum number of nearest neighbors to consider
(can be 0). This is an upper-bound, so choose a large number. If you
are unsure, you can guess and add ``check_nlist = True`` to your
constructor. This will cause the program to halt if you choose too low.
``output_forces`` indicates if the model will output forces to use in
the simulation. In the :py:meth:`compute(nlist, positions, box)<.SimModel.compute>` function you will have three
tensors that can be used:

*  ``nlist`` is an ``N`` x ``NN`` x 4 tensor containing the nearest
   neighbors. An entry of all zeros indicates that less than ``NN`` nearest
   neighbors where present for a particular particle. The 4 right-most
   dimensions are ``x,y,z`` and ``w``, which is the particle type. Particle
   type is an integer starting at 0. Note that the ``x,y,z`` values are a
   vector originating at the particle and ending at its neighbor.

* ``positions`` is an ``N`` x 4 tensor of particle positions (x,y,z) and type.

* ``box`` is a 3x3 tensor containing the low box
  coordinate (row 0), high box coordinate (row 1), and then tilt factors (row 2).

Your function can use fewer tensors, like ``compute(self, nlist)`` if
desired.

.. _Keras_Model:

Keras Model
-----------

Your models are Keras :obj:`tf.keras.Model`s so that all
the usual process of building layers, saving, and computing metrics apply. For example,
here is a two hidden layer neural network force-field that uses the 8 nearest neighbors to compute
forces.

.. code:: python

  class NlistNN(htf.SimModel):
      def setup(self, dim, top_neighs):
          self.dense1 = tf.keras.layers.Layer(dim)
          self.dense2 = tf.keras.layers.Layer(dim)
          self.last = tf.keras.layers.Layer(1)
          self.top_neighs = top_neighs

      def compute(self, nlist):
          rinv = htf.nlist_rinv(nlist)
          # closest neighbors have largest value in 1/r, take top
          top_n = tf.sort(rinv, axis=1, direction='DESCENDING')[
              :, :self.top_neighs]
          # run through NN
          x = self.dense1(top_n)
          x = self.dense2(x)
          energy = self.last(x)
          forces = htf.compute_nlist_forces(nlist, energy)
          return forces
  model = NlistNN(64, dim=16, top_neighs=8)

The ``64`` is the nlist size, ``dim`` is the hidden layer dimension, and ``top_neighs`` is how many neighbors to consider.
Refer to the Keras documentation to learn more about models.

.. _molecule_batching:

Molecule Batching
-----------------

It may be simpler to have positions or neighbor lists or forces arranged
by molecule. For example, you may want to look at only a particular bond
or subset of atoms in a molecule. To do this, you can instead subclass
:py:class:`.MolSimModel`:

.. code:: python

  import hoomd.htf as htf
  class MyModel(htf.SimModel):
    def mol_compute(self, nlist, positions, mol_nlist, mol_pos, box):
      ...
      return forces, other, important, quantities

  model = MyModel(MN, NN, mol_indices)


whose argument ``MN`` is the maximum number of atoms
in a molecule and ``mol_indices`` describes the molecules in your system as
a list of atom indices. This can be created directly from a hoomd system via :py:func:`.find_molecules`.
The ``mol_indices`` are a, possibly ragged, 2D python list where each
element in the list is a list of atom indices for a molecule. For
example, ``[[0,1], [1]]`` means that there are two molecules with the
first containing atoms 0 and 1 and the second containing atom 1. Note
that the molecules can be different size and atoms can exist in multiple
molecules.


:obj:`mol_compute(self, nlist, positions, mol_nlist, mol_pos)<.mol_compute>` has the following additional arguments:
``mol_positions`` and ``mol_nlist``. These new attributes are dimension
``M x MN x ...`` where ``M`` is the number of molecules and ``MN`` is
the atom index within the molecule. If your molecule has fewer than
``MN`` atoms, extra entries will be zeros. You can define a molecule to be
whatever you want, and atoms need not be only in one molecule. Here's an
example to compute a water angle, where we assume that the oxygens
are the middle atom:

.. code:: python

    import hoomd.htf as htf

    class MyModel(htf.SimModel):
      def mol_compute(self, nlist, positions, mol_nlist, mol_pos):
            # want slice for all molecules (:)
            # want h1 (0), o (1), h2(2)
            # positions are x,y,z,w. We only want x,y z (:3)
            v1 = mol_pos[:, 2, :3] - mol_pos[:, 1, :3]
            v2 = mol_pos[:, 0, :3] - mol_pos[:, 1, :3]
            # compute per-molecule dot product and divide by per molecule norm
            c = tf.einsum('ij,ij->i', v1, v2) / tf.norm(v1, axis=1) / tf.norm(v2 axis=1)
            angles = tf.math.acos(c)
        return angles

    # ...set-up hoomd...
    mol_indices = htf.find_molecules(hoomd_system)
    model = MyModel(3, 128, mol_indices, output_forces=False)

.. _computing_forces:

Computing Forces
----------------

If your graph is outputting forces, they must be the first return value from
your ``compute`` method. It is easiest to compute forces using
the automatic differentiation of a potential energy. Call
:py:func:`.compute_nlist_forces` with the argument ``nlist`` and ``energy``. ``energy``
can be either a scalar or a tensor which depends on ``nlist``. A tensor of
forces will be returned as :math:`\sum_i(\frac{-\partial E} {\partial n_i})`, where the sum is over
the neighbor list. For example, to compute a :math:`1 / r` potential:

.. code:: python

    import hoomd.htf as htf
    class MyModel(htf.SimModel):
      def compute(self, nlist, positions):
        #remove w since we don't care about types
        r = tf.norm(nlist[:, :, :3], axis=2)
        pairwise_energy = 0.5 * tf.math.divide_no_nan(1, r)
        # sum over neighbors
        energy = tf.reduce_sum(pairwise_energy, axis = 1)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


Notice that in the above example that we have used the
:obj:`tf.math.divide_no_nan` method, which allows
us to safely treat a :math:`1 / 0`, which can arise because ``nlist``
contains 0s for when fewer than ``NN`` nearest neighbors are found.

There is also a method :py:func:`compute_positions_forces(positions, energy)<.compute_positions_forces>` which
can be used to compute position dependent forces.

**Note:** because ``nlist`` is a *full*
neighbor list, you should divide by 2 if your energy is a sum of
pairwise energies.

.. _neighbor_lists:

Neighbor lists
--------------

The ``nlist`` is an ``N x NN x 4``
neighbor list tensor. You can compute a masked versions of this with
:py:func:`masked_nlist(nlist, type_tensor, type_i, type_j)<.masked_nlist>`
where ``type_i`` and ``type_j`` are optional integers that specify the type of
the origin (``type_i``) or neighbor (``type_j``).  ``type_tensor`` is
``positions[:,3]`` or your own types can be chosen. You can also use :py:func:`nlist_rinv(nlist)<.nlist_rinv>` which gives a
pre-computed ``1 / r`` (dimension ``N x NN``).

.. _virial:

Virial
------

A virial term can be added by doing *both* of the following extra steps:

1. Compute virial with your forces :py:func:`compute_nlist_forces(nlist, energy,virial=True)<.compute_nlist_forces>` by adding the ``virial=True`` arg.
2. Add the ``modify_virial=True`` argument to your model constructor

.. _model_saving_and_loading:


Mapped quantities
------------------

If mapped quantities are desired for coarse-graining while running a simulation, you can call
:py:meth:`tfcompute.enable_mapped_nlist(system, mapping_fxn)<.tfcompute.enable_mapped_nlist>` to utilize hoomd to compute fast neighbor lists.
The model code can then use :py:meth:`.SimModel.mapped_nlist` and
:py:meth:`.SimModel.mapped_positions` to access mapped nlist and positions. An example:

.. code:: python

  import hoomd.htf as htf

  def mapping_fxn(AA):
    return M @ AA

  class MyModel(htf.SimModel):
    def compute(self, nlist, positions, forces):
      aa_nlist, mapped_nlist = self.mapped_nlist(nlist)
      aa_pos, mapped_pos = self.mapped_positions(positions)

  ...

  tfcompute.enable_mapped_nlist(system, mapping_fxn)

Call :py:meth:`.tfcompute.enable_mapped_nlist` prior to running
the simulation.

Model Saving and Loading
---------------------------

To save a model:

.. code::python

  model.save('/path/to/save')

Because these models do not use standard Keras objects, to reload a model
you must first use your python code to build the model and then
load weights into from a file like so:

.. code:: python

  tmp_loaded_model = tf.keras.load_model('/path/to/model')
  model = MyModel(16, output_forces=True)
  model.set_weights(tmp_loaded_model.get_weights())

.. _complete_examples:

Complete Examples
-----------------

The file ``htf/test-py/build_examples.py`` contains example models

.. _lennard_jones_example:

Lennard-Jones with 1 Particle Type
----------------------------------

.. code:: python

  class LJModel(htf.SimModel):
      def compute(self, nlist):
          # get r
          rinv = htf.nlist_rinv(nlist)
          inv_r6 = rinv**6
          # pairwise energy. Double count -> divide by 2
          p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
          # sum over pairwise energy
          energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
          forces = htf.compute_nlist_forces(nlist, energy)
          return forces
