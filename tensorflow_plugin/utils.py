# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import numpy as np
from os import path
import pickle
import hoomd

def load_variables(model_directory, names, checkpoint = -1, feed_dict={}):
     # just in case
    tf.reset_default_graph()
    # load graph
    tf.train.import_meta_graph(path.join('{}/'.format(model_directory),'model.meta'), import_scope='')
    # add colons if missing
    tf_names = [n + ':0' if len(n.split(':')) == 1 else n for n in names]
    run_dict = {n:tf.get_default_graph().get_tensor_by_name(n) for n in tf_names}

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if(checkpoint == -1):
            # get latest
            checkpoint_str = model_directory
            checkpoint = tf.train.latest_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint)
            checkpoint = 'latest'
        elif type(checkpoint) == int:
            # get specific checkpoint number
            checkpoint_str = '{}/model-{}'.format(model_directory, checkpoint)
            checkpoint = tf.train.load_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint_str)
        else:
            checkpoint_str = checkpoint
            checkpoint = tf.train.load_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint_str)
        result = sess.run(run_dict, feed_dict={})
    # re add without colon if necessary
    combined_result = {}
    for k,v in result.items():
        combined_result[k] = v
        combined_result[k.split(':')[0]] = v
    return combined_result


def compute_pairwise_potential(model_directory, r, potential_tensor_name, checkpoint = -1, feed_dict = {}):
    ''' Compute the pairwise potential at r for the given model.

    Parameters
    ----------
    model_directory
        The model directory
    r
        A 1D grid of points at which to compute the potential.
    potential_tensor_name
        The tensor containing potential energy.
    checkpoint
        Which checkpoint to load. Default is -1, which loads latest checkpoint. An integer indicates loading
        from the model directory. If you pass a string, it is interpreted as a path.
    feed_dict
        Allows you to add any other placeholder values that need to be added to compute potential in your model
    Returns
    -------
    A 1D array of potentials corresponding the pairwise distances in r.
    '''
    # just in case
    tf.reset_default_graph()
    # load graph
    tf.train.import_meta_graph(path.join('{}/'.format(model_directory),'model.meta'), import_scope='')
    with open('{}/graph_info.p'.format(model_directory), 'rb') as f:
        model_params = pickle.load(f)
    if not ':' in potential_tensor_name:
        potential_tensor_name += ':0'
    potential_tensor = tf.get_default_graph().get_tensor_by_name(potential_tensor_name)
    nlist_tensor = tf.get_default_graph().get_tensor_by_name(model_params['nlist'])

    #build nlist
    NN = model_params['NN']
    np_nlist = np.zeros( (2, NN, 4) )
    potential = np.empty(len(r))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if(checkpoint == -1):
            # get latest
            checkpoint_str = model_directory
            checkpoint = tf.train.latest_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint)
            checkpoint = 'latest'
        elif type(checkpoint) == int:
            # get specific checkpoint number
            checkpoint_str = '{}/model-{}'.format(model_directory, checkpoint)
            checkpoint = tf.train.load_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint_str)
        else:
            checkpoint_str = checkpoint
            checkpoint = tf.train.load_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint_str)
        for i,ri in enumerate(r):
            np_nlist[0,0,1] = ri / 2
            np_nlist[1,0,1] = ri / 2
            # run including passed in feed_dict
            result = sess.run(potential_tensor, feed_dict = {**feed_dict, nlist_tensor: np_nlist} )
            potential[i] = result[0]
    return potential


def find_molecules(system):
    '''Given a hoomd system, this will return a mapping from molecule index to particle index

        This is a slow function and should only be called once.
    '''
    mapping = []
    mapped = set()
    N = len(system.particles)
    unmapped = set(range(N))
    pi = 0
    while len(mapped) != N:
        pi = unmapped.pop()
        mapped.add(pi)
        mapping.append([pi])
        # traverse bond group
        # until no more found
        keep_going = True
        while keep_going:
            for bond in system.bonds:
                # see if bond contains pi and an unseen atom
                if (pi == bond.a and bond.b in unmapped) or \
                    (pi == bond.b and bond.a in unmapped):
                    pi = bond.a if pi == bond.b else bond.b
                    unmapped.remove(pi)
                    mapped.add(pi)
                    mapping[-1].append(pi)
                    keep_going = True
                    break
                keep_going = False
    # sort it to be ascending in min atom index in molecule
    for m in mapping:
        m.sort()
    mapping.sort(key=lambda x: min(x))
    return mapping

def sparse_mapping(molecule_mapping, molecule_mapping_index, system=None):
    ''' This will create the necessary indices and values for defining a sparse tensor in
    tensorflow that is a mass-weighted M x N mapping operator.

    Parameters
    -----------
    molecule_mapping
        This is a list of L x M matrices, where M is the number of atoms in the molecule and L
        is the number of coarse-grain sites that should come out of the mapping.
        There should be one matrix per molecule. The ordering of the atoms should follow
        what is defined in the output from find_molecules
    molecule_mapping_index
        This is the output from find_molecules.
    system
        The hoomd system. This is used to get mass values for the mapping, if you would like to
        weight by mass of the atoms.
    Returns
    -------
        A sparse tensorflow tensor of dimension N x N, where N is number of atoms
    '''
    import numpy as np
    assert type(molecule_mapping[0]) == np.ndarray

    # get system size
    N = sum([len(m) for m in molecule_mapping_index])
    M = sum([m.shape[0] for m in molecule_mapping])

    # create indices
    indices = []
    values = []
    total_i = 0
    for mmi,mm in zip(molecule_mapping_index, molecule_mapping):
        idx = []
        vs = []
        masses = [0 for _ in range(mm.shape[0])]
        # iterate over CG particles
        for i in range(mm.shape[0]):
            # iterate over atoms
            for j in range(mm.shape[1]):
                # check if non-zero
                if mm[i, j] > 0:
                    # index -> CG particle, atom index
                    idx.append([i + total_i, mmi[j]])
                    if system is not None:
                        vs.append(system.particles[mmi[j]].mass)
                    else:
                        vs.append(1)
        # now add up masses
        for i in range(len(idx)):
            # get masses from previous values
            masses[idx[i][0] - total_i] += vs[i]
        # make sure things are valid
        assert sum([m == 0 for m in masses]) == 0
        # now scale values by mases
        for i in range(len(idx)):
            vs[i] /= masses[idx[i][0] - total_i]
        # all donw
        indices.extend(idx)
        values.extend(vs)
        total_i += len(masses)

    return tf.SparseTensor(indices=indices, values=values, dense_shape=[M, N])
