# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White
import tensorflow as tf
import os
import pickle


class SimModel(tf.keras.Model):
    R'''
    SimModel is the main way that Hoomd-TF interacts with a simulation.
    '''

    def __init__(
            self, nneighbor_cutoff, output_forces=True,
            virial=False, check_nlist=False, dtype=tf.float32,
            name='htf-model', **kwargs):
        R'''

            SimModel is the main way that Hoomd-TF interacts with a simulation.
            Any ``kwargs`` are passed to :py:meth:`setup`.

            :param nneighbor_cutoff: The maximum number of neighbors to consider (can be 0)
            :type nneighbor_cutoff: int
            :param output_forces: True if your graph will compute
                forces to be used in the Hoomd simulation
            :type output_forces: bool
            :param check_nlist: True will raise error if neighbor
                                list overflows (nneighbor_cutoff too low)
            :type check_nlist: bool
            :param dtype: The floating point specification for model (e.g., ``tf.float32``)
            :type dtype: dtype

            '''

        super(SimModel, self).__init__(dtype=dtype, name=name)
        self.nneighbor_cutoff = nneighbor_cutoff
        self.output_forces = output_forces
        self.virial = virial

        input_signature = [
            tf.TensorSpec(
                shape=[None, max(1, nneighbor_cutoff), 4], dtype=dtype),  # nlist
            tf.TensorSpec(shape=[None, 4], dtype=dtype),  # positions
            tf.TensorSpec(shape=[None, 3], dtype=dtype),  # box
            tf.TensorSpec(shape=[])  # batch_frac (sample weight)
        ]

        try:
            self.compute = tf.function(
                self.compute, input_signature=input_signature)
        except AttributeError:
            raise AttributeError(
                'SimModel child class must implement compute method, and should not implement call')

        self.check_nlist = check_nlist
        self.batch_steps = tf.Variable(
            name='htf-batch-steps', dtype=tf.int32, initial_value=0, trainable=False)
        self._running_means = []
        self.setup(**kwargs)

    def compute(self, nlist, positions, box, sample_weight):
        R'''
        The main method were computation occurs occurs. This method must be implemented
        by subclass. It should return one or more values as a tuple.
        The first element is interpreted as forces (if ``output_forces=True``, default).
        Second element is interpreted as virial (if ``virial=True``, not default). Subsequent
        elements of tuple are only accessible if :py:meth:`.tfcompute.attach` is passed
        ``save_output_period``, after which you can obtain from the ``tfcompute`` object
        as the ``outputs`` attribute. Use :py:func:`compute_nlist_forces` or
        :py:func:`compute_positions_forces` to compute forces from an energy.

        :param nlist: an ``N x NN x 4`` tensor containing the nearest
            neighbors. An entry of all zeros indicates that less than ``NN`` nearest
            neighbors where present for a particular particle. The last axis 4
            dimensions are ``x,y,z`` and ``w``, which is the particle type. Particle
            type is an integer starting at 0. Note that the ``x,y,z`` values are a
            vector originating at the particle and ending at its neighbor.
        :type nlist: tensor
        :param positions: an ``N x 4`` tensor of particle positions (x,y,z) and type.
        :type positions: tensor

        :param box: a ``3x3`` tensor containing the low box coordinate (row 0),
            high box coordinate (row 1), and then tilt factors (row 2).
            Call :py:func:`.box_size` to convert to size
        :type box: tensor

        :param sample_weight: a floating point fraction indicating
            what amount of the total system is being passed in this call.
            Is only NOT 1.0 if ``batch_size`` is passed to :py:meth:`.tfcompute.attach`.
        :type sample_weight: tensor

        :return: Tuple of tensors

        '''
        raise AttributeError(
            'You must implement compute in your subclass')

    def setup(self, **kwargs):
        R'''
        This method can be implemented by a subclass to perform
        tasks after object creation. Any ``kwargs`` passed to
        :py:meth:`.SimModel.__init__` will be args
        here. This method will be called automatically.
        '''
        pass

    def call(self, inputs, training):
        out = self.compute(*inputs)
        if tf.is_tensor(out):
            out = (out,)
        return out

    @tf.function
    def compute_inputs(self, dtype, nlist_addr, positions_addr,
                       box_addr, batch_frac, forces_addr=0):
        hoomd_to_tf_module = load_htf_op_library('hoomd2tf_op')
        hoomd_to_tf = hoomd_to_tf_module.hoomd_to_tf

        if self.nneighbor_cutoff > 0:
            nlist = tf.reshape(hoomd_to_tf(
                address=nlist_addr,
                shape=[4 * self.nneighbor_cutoff],
                T=dtype,
                name='nlist-input'
            ), [-1, self.nneighbor_cutoff, 4])
        else:
            nlist = tf.zeros([1, 1, 4], dtype=self.dtype)
        pos = hoomd_to_tf(
            address=positions_addr,
            shape=[4],
            T=dtype,
            name='pos-input'
        )

        box = hoomd_to_tf(
            address=box_addr,
            shape=[3],
            T=dtype,
            name='box-input'
        )

        # check box skew
        tf.Assert(tf.less(tf.reduce_sum(box[2]), 0.0001), ['box is skewed'])

        if self.check_nlist:
            NN = tf.reduce_max(
                input_tensor=tf.reduce_sum(
                    input_tensor=tf.cast(
                        nlist[:, :, 0] > 0,
                        tf.dtypes.int32), axis=1),
                axis=0)
            tf.debugging.assert_less(NN, self.nneighbor_cutoff,
                                     message='Neighbor list is full!')

        result = [tf.cast(nlist, self.dtype), tf.cast(
            pos, self.dtype), tf.cast(box, self.dtype), batch_frac]

        if forces_addr > 0:
            forces = hoomd_to_tf(
                address=forces_addr,
                shape=[4],
                T=dtype,
                name='forces-input'
            )
            result.append(tf.cast(forces, self.dtype))

        return result

    @tf.function
    def compute_outputs(self, dtype, force_addr, virial_addr, forces, virial=None):

        if forces.shape[1] == 3:
            forces = tf.concat(
                [forces, tf.zeros(tf.shape(forces)[0])[:, tf.newaxis]],
                axis=1, name='forces')
        tf_to_hoomd_module = load_htf_op_library('tf2hoomd_op')
        tf_to_hoomd = tf_to_hoomd_module.tf_to_hoomd
        tf_to_hoomd(
            tf.cast(forces, dtype),
            address=force_addr)
        if virial is not None:
            tf_to_hoomd(
                tf.cast(virial, dtype),
                address=virial_addr)


class MolSimModel(SimModel):
    '''
    A molecular batched py:class:`.SimModel`
    '''

    def __init__(
            self, MN, mol_indices,
            nneighbor_cutoff, output_forces=True,
            virial=False, check_nlist=False,
            dtype=tf.float32, name='htf-mol-model',
            **kwargs):
        R'''
        The change from :py:meth:`.SimModel.__init__` are the first two parameters.

        :param MN: The number of atoms in a molecule.
            MN must be chosen to be large enough to encompass all molecules. If your molecule
            is 6 atoms and you chose MN=18, then the extra entries will be zeros.
        :type MN: int

        :param mol_indices: ``mol_indices`` describes the molecules in your system as
            a list of atom indices. This can be created directly from a
            hoomd system via :py:func:`.find_molecules`.
            The ``mol_indices`` are a, possibly ragged, 2D python list where each
            element in the list is a list of atom indices for a molecule. For
            example, ``[[0,1], [1]]`` means that there are two molecules with the
            first containing atoms 0 and 1 and the second containing atom 1. Note
            that the molecules can be different size and atoms can exist in multiple
            molecules.
        :type mol_indices: list of lists
        '''

        super(MolSimModel, self).__init__(
            nneighbor_cutoff, output_forces=output_forces,
            virial=virial, check_nlist=check_nlist,
            dtype=dtype, name=name, **kwargs)
        self.MN = MN

        self.mol_indices = mol_indices

        # fill out the indices
        for mi in self.mol_indices:
            for i in range(len(mi)):
                # add 1 so that an index of 0 corresponds to slicing a dummy atom
                mi[i] += 1
            if len(mi) > MN:
                raise ValueError('One of your molecule indices'
                                 ' has more than MN indices.'
                                 'Increase MN in your graph.')
            while len(mi) < MN:
                mi.append(0)

        self.rev_mol_indices = _make_reverse_indices(mol_indices)

        input_signature = [
            tf.TensorSpec(
                shape=[None, MN, max(1, nneighbor_cutoff), 4], dtype=dtype),  # nlist
            tf.TensorSpec(shape=[None, MN, 4], dtype=dtype),  # positions
            tf.TensorSpec(
                shape=[None, MN, max(1, nneighbor_cutoff), 4], dtype=dtype),  # mol_nlist
            tf.TensorSpec(shape=[None, MN, 4], dtype=dtype),  # mol_positions
            tf.TensorSpec(shape=[None, 3], dtype=dtype),  # box
            tf.TensorSpec(shape=[])  # batch_frac (sample weight)
        ]
        try:
            self.mol_compute
        except AttributeError:
            raise AttributeError(
                'SimModel child class must implement compute method, '
                'and should not implement call')

    def mol_compute(self, nlist, positions, mol_nlist, mol_positions, box):
        R'''
        See :py:meth:`.SimModel.compute` for details. ``sample_weight``
        is not passed because simulations cannot currently be batched both by size and molecule.
        Make sure that your forces still use ``nlist`` when computing, instead of ``mol_nlist``.

        :param nlist: an ``N x NN x 4`` tensor containing the nearest
            neighbors. An entry of all zeros indicates that less than ``NN`` nearest
            neighbors where present for a particular particle. The last axis 4
            dimensions are ``x,y,z`` and ``w``, which is the particle type. Particle
            type is an integer starting at 0. Note that the ``x,y,z`` values are a
            vector originating at the particle and ending at its neighbor.
        :type nlist: tensor
        :param positions: an ``N x 4`` tensor of particle positions (x,y,z) and type.
        :type positions: tensor

        :param mol_nlist: a ``mol_number x MN x NN x 4`` tensor containing the nearest
            neighbors broken out by molecule. An entry of all zeros indicates
            that less than ``NN`` nearest
            neighbors where present for a particular particle. The last axis 4
            dimensions are ``x,y,z`` and ``w``, which is the particle type. Particle
            type is an integer starting at 0. Note that the ``x,y,z`` values are a
            vector originating at the particle and ending at its neighbor.
        :type nlist: tensor
        :param mol_positions: a ``mol_number x MN x N x 4`` tensor of
            particle positions (x,y,z) and type.
        :type positions: tensor

        :param box: a ``3x3`` tensor containing the low box coordinate (row 0),
            high box coordinate (row 1), and then tilt factors (row 2).
            Call :py:func:`.box_size` to convert to size
        :type box: tensor

        :return: Tuple of tensors

        '''
        raise AttributeError('You must implement mol_compute method')

    def compute(self, nlist, positions, box, batch_frac):

        mol_flat_idx = tf.reshape(self.mol_indices, shape=[-1])

        # we add one dummy particle to the positions, nlist, and forces so that
        # we can fill the mol indices with 0s which will slice
        # these dummy particles. Thus we will add one to the mol indices when
        # we do tf compute to prepare.
        ap = tf.concat((
            tf.constant([0, 0, 0, 0], dtype=positions.dtype,
                        shape=(1, 4)),
            positions),
            axis=0)
        an = tf.concat(
            (tf.zeros(shape=(1, self.nneighbor_cutoff, 4),
                      dtype=positions.dtype), nlist),
            axis=0)
        mol_positions = tf.reshape(
            tf.gather(ap, mol_flat_idx), shape=[-1, self.MN, 4])
        mol_nlist = tf.reshape(
            tf.gather(an, mol_flat_idx),
            shape=[-1, self.MN, self.nneighbor_cutoff, 4])
        return self.mol_compute(nlist, positions, mol_nlist, mol_positions, box)


def compute_positions_forces(positions, energy):
    R'''
    Computes position dependent forces given
    a potential energy function. Returns forces as a ``N x 4`` tensor, where
    the last dimension of axis 1 is per-particle energy if available, otherwise
    0.

    :param positions: ``N x 4`` or ``N x 3`` positions tensor
    :type nlist: tensor
    :param energy: The potential energy. Can be size ``1``, ``N``, or ``N x ?``
    :type energy: tensor
    :return: Forces as tensor
    '''
    forces = tf.gradients(energy, positions)[0]
    return _add_energy(forces, energy)


def _compute_virial(nlist, nlist_forces):
    # now treat virial
    nlist3 = nlist[:, :, :3]
    rij_outter = tf.einsum('ijk,ijl->ijkl', nlist3, nlist3)
    # F / rs
    nlist_r_mag = tf.norm(
        nlist3, axis=2, name='nlist-r-mag')
    nlist_force_mag = tf.norm(
        nlist_forces, axis=2, name='nlist-force-mag')
    F_rs = tf.math.divide_no_nan(nlist_force_mag, 2.0 *
                                 nlist_r_mag)
    # sum over neighbors: F / r * (r (outter) r)
    virial = -1.0 * tf.einsum('ij,ijkl->ikl',
                              F_rs, rij_outter)
    return virial


def compute_nlist_forces(nlist, energy, virial=False):
    R'''
    Computes pairwise forces given
    a potential energy function that computes per-particle
    or overall energy. Returns forces as a ``N x 4`` tensor, where
    the last dimension of axis 1 is per-particle energy if available, otherwise
    0.

    :param nlist: ``N x NN x 4`` or ``N`` x ``NN`` x 3 neighbor list
    :type nlist: tensor
    :param energy: The potential energy. Can be size ``1`` ``N`` or ``N x L``
    :type energy: tensor
    :param virial: True if virial contribution will be computed.
    :type virial: bool
    :return: Either the forces or a tuple with forces, virial.
    '''
    nlist_grad = tf.gradients(energy, nlist)[0]
    if nlist_grad is None:
        raise ValueError(
            'Could not find dependence between energy and nlist.'
            ' Did you put them in wrong order?')
    # remove 0s and *2
    nlist_forces = tf.math.multiply_no_nan(nlist_grad, 2.0)
    nlist_reduce = tf.reduce_sum(input_tensor=nlist_forces, axis=1,
                                 name='nlist-force-gradient')
    if virial:
        return tf.tuple([_add_energy(nlist_reduce, energy),
                         _compute_virial(nlist, nlist_forces)])
    else:
        return _add_energy(nlist_reduce, energy)


def _add_energy(forces, energy):
    if len(energy.shape) > 1:
        # reduce energy to be correct shape
        print('WARNING: Your energy is multidimensional per particle.'
              'Hopefully that is intentional')
        energy = tf.reshape(
            tf.reduce_sum(energy, axis=list(range(1, len(energy.shape)))),
            [tf.shape(forces)[0], 1])
        forces = tf.concat([forces[:, :3], energy], -1)
    elif len(energy.shape) == 0:
        forces = tf.concat([forces[:, :3],
                            tf.reshape(tf.tile(tf.reshape(energy, [1]),
                                               tf.shape(forces)[0:1]),
                                       shape=[-1, 1])],
                           -1)
    else:
        forces = tf.concat(
            [forces[:, :3], tf.reshape(
                energy,
                [tf.shape(forces)[0], 1])], -1)
    return forces


@tf.function
def safe_norm(tensor, delta=1e-7, **kwargs):
    '''
    There are some numerical instabilities that can occur during learning
    when gradients are propagated. The delta is problem specific.
    **Note you should not take a safe norm and then pass to** ``tf.math.divide_no_nan``
    See `this TensorFlow issue <https://github.com/tensorflow/tensorflow/issues/12071>`_.

    :param tensor: the tensor over which to take the norm
    :param delta: small value to add so near-zero is treated without too much
        accuracy loss.
    :return: The safe norm op (TensorFlow operation)
    '''
    return tf.norm(tensor=tensor + delta, **kwargs)


@tf.function
def wrap_vector(r, box):
    '''Computes the minimum image version of the given vector.

        :param r: The vector to wrap around the Hoomd box.
        :type r: tensor
        :return: The wrapped vector as a TF tensor
    '''
    box_size = box[1, :] - box[0, :]
    return r - tf.math.round(r / box_size) * box_size


@tf.function
def box_size(box):
    return box[1, :] - box[0, :]


@tf.function
def nlist_rinv(nlist):
    ''' Returns an ``N x NN`` tensor of 1 / r for each neighbor
    while correctly treating zeros.
    '''
    r = tf.norm(nlist[:, :, :3], axis=2)
    return tf.math.divide_no_nan(1.0, r)


@tf.function
def compute_rdf(nlist, type_tensor, r_range, nbins=100, type_i=None, type_j=None):
    '''Computes the pairwise radial distribution function

    :param nlist: Neighbor list to use for RDF calculation.
    :type nlist: tensor
    :param type_tensor: ``N x 1`` tensor containing types. Can use ``positions[:, 3]``
    :type type_tensor: tensor
    :param r_range: A list containing two elements, begin and end, for r range.
    :type r_range: 2 element list
    :param bins: The bins to use for the RDF
    :type bins: int
    :param type_i: Use this to select the first particle type.
    :type type_i: int
    :param type_j: Use this to select the second particle type.
    :type type_j: int

    :return: length ``nbins`` tensor of the RDF (not normalized).
    '''
    # to prevent type errors later on
    r_range = [float(r) for r in r_range]
    # filter types
    nlist = masked_nlist(nlist, type_tensor, type_i, type_j)
    r = tf.norm(tensor=nlist[:, :, :3], axis=2)
    hist = tf.cast(tf.histogram_fixed_width(r, r_range, nbins + 2),
                   tf.float32)
    shell_rs = tf.linspace(r_range[0], r_range[1], nbins + 1)
    vis_rs = tf.multiply((shell_rs[1:] + shell_rs[:-1]), 0.5)
    vols = shell_rs[1:]**3 - shell_rs[:-1]**3
    # remove 0s and Ns
    result = hist[1:-1] / vols
    return result, vis_rs


@tf.function
def masked_nlist(nlist, type_tensor, type_i=None, type_j=None):
    '''Returns a neighbor list masked by the given particle type(s).

    :param nlist: Neighbor list to use for RDF calculation.
    :type nlist: tensor
    :param type_tensor: ``N x 1`` tensor containing types. Can use ``positions[:, 3]``
    :type type_tensor: tensor
    :param type_i: Use this to select the first particle type.
    :type type_i: int
    :param type_j: Use this to select the second particle type.
    :type type_j: int
    :return: The masked neighbor list tensor.
    '''
    if type_i is not None:
        nlist = tf.boolean_mask(
            tensor=nlist, mask=tf.equal(type_tensor, type_i))
    if type_j is not None:
        # cannot use boolean mask due to size
        mask = tf.cast(tf.equal(nlist[:, :, 3], type_j), tf.float32)
        nlist = nlist * mask[:, :, tf.newaxis]
    return nlist


def load_htf_op_library(op):
    import hoomd.htf
    path = hoomd.htf.__path__[0]
    try:
        op_path = os.path.join(path, op, 'lib_{}'.format(op))
        if os.path.exists(op_path + '.so'):
            op_path += '.so'
        elif os.path.exists(op_path + '.dylib'):
            op_path += '.dylib'
        else:
            raise OSError()
        mod = tf.load_op_library(op_path)
    except OSError:
        raise OSError('Unable to load OP {}. '
                      'Expected to be in {}'.format(op, path))
    return mod


def _make_reverse_indices(mol_indices):
    num_atoms = 0
    for m in mol_indices:
        num_atoms = max(num_atoms, max(m))
    # you would think add 1, since we found the largest index
    # but the atoms are 1-indexed to distinguish between
    # the "no atom" case (hence the - 1 below)
    rmi = [[] for _ in range(num_atoms)]
    for i in range(len(mol_indices)):
        for j in range(len(mol_indices[i])):
            index = mol_indices[i][j]
            if index > 0:
                rmi[index - 1] = [i, j]
    warned = False
    for r in rmi:
        if len(r) != 2 and not warned:
            warned = True
            print('Not all of your atoms are in a molecule\n')
            r.extend([-1, -1])
    return rmi
