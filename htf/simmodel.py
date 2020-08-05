# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import os
import pickle


class SimModel(tf.keras.Model):
    def __init__(self, nneighbor_cutoff, output_forces=True, check_nlist=False, dtype=tf.float32, name='htf-model', **kwargs):
        R""" Build the TensorFlow graph that will be used during the HOOMD run.
        """
        super(SimModel, self).__init__(dtype=dtype, name=name, **kwargs)
        print('Building SIM MODEL!!')
        self.nneighbor_cutoff = nneighbor_cutoff
        self.output_forces = output_forces

        input_signature = [
            tf.TensorSpec(
                shape=[None, max(1, nneighbor_cutoff), 4], dtype=dtype),  # nlist
            tf.TensorSpec(shape=[None, 4], dtype=dtype),  # positions
            tf.TensorSpec(shape=[None, 3], dtype=dtype),  # box
            tf.TensorSpec(shape=[]) # batch_frac (sample weight)
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
        self.setup()

    def setup(self):
        pass

    def call(self, inputs, training):
        return self.compute(*inputs)

    @tf.function
    def compute_inputs(self, dtype, nlist_addr, positions_addr, box_addr, batch_frac, forces_addr=0):
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
            NN = tf.reduce_max(input_tensor=tf.reduce_sum(input_tensor=tf.cast(self.nlist[:, :, 0] > 0,
                                                                               tf.dtypes.int32), axis=1),
                               axis=0)
            tf.Assert(tf.less(NN, self.nneighbor_cutoff),
                      ['Neighbor list is full!'])

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
    def compute_outputs(self, dtype, force_addr, forces):
        if forces.shape[1] == 3:
            forces = tf.concat(
                [forces, tf.zeros(tf.shape(forces)[0])[:, tf.newaxis]],
                axis=1, name='forces')
        tf_to_hoomd_module = load_htf_op_library('tf2hoomd_op')
        tf_to_hoomd = tf_to_hoomd_module.tf_to_hoomd
        forces = tf_to_hoomd(
            tf.cast(forces, dtype),
            address=force_addr)

    def masked_nlist(self, type_i=None, type_j=None, nlist=None,
                     type_tensor=None):
        R"""Returns a neighbor list masked by the given particle type(s).

            :param type_i: Use this to select the first particle type.
            :param type_j: Use this to select a second particle type (optional).
            :param nlist: Neighbor list to mask. By default it will use ``self.nlist``.
            :param type_tensor: An N x 1 tensor containing the type(s) of the nlist origin.
                If None, particle types from ``self.positions`` will be used.
            :return: The masked neighbor list tensor.
        """
        if nlist is None:
            nlist = self.nlist
        if type_tensor is None:
            type_tensor = self.positions[:, 3]
        if type_i is not None:
            nlist = tf.boolean_mask(
                tensor=nlist, mask=tf.equal(type_tensor, type_i))
        if type_j is not None:
            # cannot use boolean mask due to size
            mask = tf.cast(tf.equal(nlist[:, :, 3], type_j), tf.float32)
            nlist = nlist * mask[:, :, tf.newaxis]
        return nlist

    def compute_rdf(self, r_range, name, nbins=100, type_i=None, type_j=None,
                    nlist=None, positions=None):
        R"""Computes the pairwise radial distribution function, and appends
            the histogram tensor to the graph's ``out_nodes``.

        :param bins: The bins to use for the RDF
        :param name: The name of the tensor containing rdf. The name will be
            concatenated with '-r' to create a tensor containing the
            r values of the rdf.
        :param type_i: Use this to select the first particle type.
        :param type_j: Use this to select the second particle type.
        :param nlist: Neighbor list to use for RDF calculation. By default
            it will use ``self.nlist``.
        :param positions: Defaults to ``self.positions``. This tensor is only used
            to get the origin particle's type. So if you're making your own,
            just make sure column 4 has the type index.

        :return: Historgram tensor of the RDF (not normalized).
        """
        # to prevent type errors later on
        r_range = [float(r) for r in r_range]
        if nlist is None:
            nlist = self.nlist
        if positions is None:
            positions = self.positions
        # filter types
        nlist = self.masked_nlist(type_i, type_j, nlist, positions[:, 3])
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        hist = tf.cast(tf.histogram_fixed_width(r, r_range, nbins + 2),
                       tf.float32)
        shell_rs = tf.linspace(r_range[0], r_range[1], nbins + 1)
        vis_rs = tf.multiply((shell_rs[1:] + shell_rs[:-1]), 0.5,
                             name=name + '-r')
        vols = shell_rs[1:]**3 - shell_rs[:-1]**3
        # remove 0s and Ns
        result = hist[1:-1] / vols
        self.out_nodes.extend([result, vis_rs])
        return result

    def save_tensor(self, tensor, name, save_period=1):
        R"""Saves a tensor to a variable

        :param tensor: The tensor to save
        :type tensor: tensor
        :param name: The name of the variable which will be saved
        :type name: str
        :param save_period: How often to save the variable
        :type save_period: int

        :return: None
        """

        # make sure it is a tensor
        if type(tensor) != tf.Tensor:
            raise ValueError('save_tensor requires a tf.Tensor '
                             'but given type {}'.format(type(tensor)))

        store = tf.Variable(name=name, initial_value=tf.zeros_like(tensor),
                            validate_shape=False, dtype=tensor.dtype, trainable=False)

        store_op = store.assign(tensor)
        self.out_nodes.append([store_op, save_period])

    def build_mol_rep(self, MN):
        R"""
        This creates ``mol_forces``, ``mol_positions``, and ``mol_nlist`` which have dimensions
        mol_number x MN x 4 (``mol_forces``, ``mol_positions``) and
        ? x MN x NN x 4 (``mol_nlist``) tensors batched by molecule, where MN
        is the number of molecules. MN is determined at run time. The MN must
        be chosen to be large enough to encompass all molecules. If your molecule
        is 6 atoms and you chose MN=18, then the extra entries will be zeros. Note
        that your input should be 0 based, but subsequent tensorflow data will be 1 based,
        since 0 means no atom. The specification of what is a molecule
        will be passed at runtime, so that it can be dynamic if desired.

        To convert a mol_quantity to a per-particle quantity, call
        ``scatter_mol_quanitity(mol_quantity)``

        :param MN: The number of molecules
        :type MN: int
        :return: None
        """

        self.mol_indices = tf.compat.v1.placeholder(tf.int32,
                                                    shape=[None, MN],
                                                    name='htf-molecule-index')

        self.rev_mol_indices = tf.compat.v1.placeholder(tf.int32,
                                                        shape=[None, 2],
                                                        name='htf-reverse-molecule-index')
        self.mol_flat_idx = tf.reshape(self.mol_indices, shape=[-1])

        # we add one dummy particle to the positions, nlist, and forces so that
        # we can fill the mol indices with 0s which will slice
        # these dummy particles. Thus we will add one to the mol indices when
        # we do tf compute to prepare.
        ap = tf.concat((
            tf.constant([0, 0, 0, 0], dtype=self.positions.dtype,
                        shape=(1, 4)),
            self.positions),
            axis=0)
        an = tf.concat(
            (tf.zeros(shape=(1, self.nneighbor_cutoff, 4),
                      dtype=self.positions.dtype), self.nlist),
            axis=0)
        self.mol_positions = tf.reshape(
            tf.gather(ap, self.mol_flat_idx), shape=[-1, MN, 4])
        self.mol_nlist = tf.reshape(
            tf.gather(an, self.mol_flat_idx),
            shape=[-1, MN, self.nneighbor_cutoff, 4])
        if not self.output_forces:
            af = tf.concat((
                tf.constant(
                    [0, 0, 0, 0], dtype=self.positions.dtype, shape=(1, 4)),
                self.forces),
                axis=0)
            self.mol_forces = tf.reshape(
                tf.gather(af, self.mol_flat_idx), shape=[-1, 4])
        self.MN = MN


def compute_positions_forces(positions, energy):
    forces = tf.gradients(energy, positions)[0]
    return add_energy(forces, energy)


def compute_nlist_forces(nlist, energy):
    nlist_grad = tf.gradients(energy, nlist)[0]
    if nlist_grad is None:
        raise ValueError('Could not find dependence between energy and nlist')
    nlist_grad = tf.identity(tf.math.multiply(tf.constant(2.0), nlist_grad),
                             name='nlist-pairwise-force'
                             '-gradient-raw')
    zeros = tf.zeros(tf.shape(input=nlist_grad))
    nlist_forces = tf.compat.v1.where(tf.math.is_finite(nlist_grad),
                                      nlist_grad, zeros,
                                      name='nlist-pairwise-force-gradient')
    nlist_reduce = tf.reduce_sum(input_tensor=nlist_forces, axis=1,
                                 name='nlist-force-gradient')
    return add_energy(nlist_reduce, energy)

def add_energy(forces, energy):
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
    R"""
    There are some numerical instabilities that can occur during learning
    when gradients are propagated. The delta is problem specific.
    NOTE: delta of tf.math.divide_no_nan must be > sqrt(3) * (safe_norm delta)
    See `this TensorFlow issue <https://github.com/tensorflow/tensorflow/issues/12071>`.

    :param tensor: the tensor over which to take the norm
    :param delta: small value to add so near-zero is treated without too much
        accuracy loss.
    :return: The safe norm op (TensorFlow operation)
    """
    return tf.norm(tensor=tensor + delta, **kwargs)


@tf.function
def wrap_vector(r, box):
    R"""Computes the minimum image version of the given vector.

        :param r: The vector to wrap around the HOOMD box.
        :type r: tensor
        :return: The wrapped vector as a TF tensor
    """
    box_size = box[1, :] - box[0, :]
    return r - tf.math.round(r / box_size) * box_size

@tf.function
def nlist_rinv(nlist):
    R""" Returns an N x NN tensor of 1 / r for each neighbor
    """
    r = tf.norm(nlist[:, :, :3], axis=2)
    return tf.math.divide_no_nan(1.0, r)

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