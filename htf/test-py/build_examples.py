# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import os
import hoomd.htf as htf
import pickle


def simple_potential(directory='/tmp/test-simple-potential-model'):
    graph = htf.graph_builder(9 - 1)
    with tf.name_scope('force-calc') as scope:
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(nlist, axis=2, keepdims=True)
        # no need to use netwon's law because nlist should be double counted
        fr = tf.multiply(-1.0, tf.multiply(tf.reciprocal(neighs_rs), nlist),
                         name='nan-pairwise-forces')
        with tf.name_scope('remove-nans') as scope:
            zeros = tf.zeros_like(nlist)
            real_fr = tf.where(tf.is_finite(fr), fr, zeros,
                               name='pairwise-forces')
        forces = tf.reduce_sum(real_fr, axis=1, name='forces')
    graph.save(force_tensor=forces, model_directory=directory)
    return directory
    # check graph info
    with open('/tmp/test-simple-potential-model/graph_info.p', 'rb') as f:
        gi = pickle.load(f)
        assert gi['forces'] != 'forces:0'
        assert tf.get_default_graph().get_tensor_by_name(gi['forces'
                                                            ]).shape[1] == 4


def benchmark_gradient_potential():
    graph = htf.graph_builder(1024, 64)
    nlist = graph.nlist[:, :, :3]
    # get r
    r = tf.norm(nlist, axis=2)
    # compute 1 / r while safely treating r = 0.
    energy = tf.reduce_sum(graph.safe_div(1., r), axis=1)
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces,
               model_directory='/tmp/benchmark-gradient-potential-model')


def gradient_potential():
    graph = htf.graph_builder(9 - 1)
    with tf.name_scope('force-calc') as scope:
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(nlist, axis=2)
        energy = 0.5 * graph.safe_div(numerator=tf.ones_like(
                neighs_rs, dtype=neighs_rs.dtype), denominator=neighs_rs,
                                      name='energy')
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces,
               model_directory='/tmp/test-gradient-potential-model',
               out_nodes=[energy])


def noforce_graph(directory='/tmp/test-noforce-model'):
    graph = htf.graph_builder(9 - 1, output_forces=False)
    nlist = graph.nlist[:, :, :3]
    neighs_rs = tf.norm(nlist, axis=2)
    energy = graph.safe_div(numerator=tf.ones_like(
            neighs_rs, dtype=neighs_rs.dtype),
                            denominator=neighs_rs, name='energy')
    pos_norm = tf.norm(graph.positions, axis=1)
    graph.save(directory, out_nodes=[energy, pos_norm])
    return directory


def wrap_graph(directory='/tmp/test-wrap-model'):
    graph = htf.graph_builder(0, output_forces=False)
    p1 = graph.positions[0, :3]
    p2 = graph.positions[-1, :3]
    r = p1 - p2
    rwrap = graph.wrap_vector(r)
    # TODO: Smoke test. Think of a better test.
    graph.save(directory, out_nodes=[rwrap])
    return directory

def mol_force(directory='/tmp/test-mol-force-model'):
    graph = htf.graph_builder(0, output_forces=False)
    graph.build_mol_rep(3)
    f = tf.norm(graph.mol_forces, axis=0)
    graph.save(directory, out_nodes=[f])
    return directory


def feeddict_graph(directory='/tmp/test-feeddict-model'):
    graph = htf.graph_builder(9 - 1, output_forces=False)
    forces = graph.forces[:, :3]
    force_com = tf.reduce_mean(forces, axis=0)
    thing = tf.placeholder(dtype=tf.float32, name='test-tensor')
    out = force_com * thing
    graph.save(directory, out_nodes=[out])
    return directory


def benchmark_nonlist_graph(directory='/tmp/benchmark-nonlist-model'):
    graph = htf.graph_builder(0, output_forces=True)
    ps = tf.norm(graph.positions, axis=1)
    energy = graph.safe_div(1., ps)
    force = graph.compute_forces(energy)
    graph.save(directory, force_tensor=force, out_nodes=[energy])
    return directory


def lj_graph(NN, directory='/tmp/test-lj-potential-model'):
    graph = htf.graph_builder(NN)
    nlist = graph.nlist[:, :, :3]
    # get r
    r = tf.norm(nlist, axis=2)
    # compute 1 / r while safely treating r = 0.
    # pairwise energy. Double count -> divide by 2
    inv_r6 = graph.safe_div(1., r**6)
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    # sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    # compute energy every 10 steps
    graph.save(force_tensor=forces, model_directory=directory, out_nodes=[[energy, 10]])
    return directory


def eds_graph(directory='/tmp/test-lj-eds'):
    graph = htf.graph_builder(0)
    # get distance from center
    rvec = graph.positions[0,:3]
    cv = tf.norm(rvec)
    alpha = htf.eds_bias(cv, 2, 5)
    energy = cv * alpha
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory=directory, out_nodes=[alpha])
    return directory


def custom_nlist(NN, r_cut, system, directory='/tmp/test-custom-nlist'):
    graph = htf.graph_builder(NN, output_forces=False)
    nlist = graph.nlist[:, :, :3]
    # get r
    r = tf.norm(nlist, axis=2)
    v = tf.get_variable('hoomd-r', initializer=tf.zeros_like(r),
                        validate_shape=False)
    ops = [v.assign(r)]

    # compute nlist
    cnlist = htf.compute_nlist(graph.positions[:, :3], r_cut, NN, system)
    r = tf.norm(cnlist[:, :, :3], axis=2)
    v = tf.get_variable('htf-r', initializer=tf.zeros_like(r),
                        validate_shape=False)
    ops.append(v.assign(r))

    graph.save(model_directory=directory, out_nodes=ops)
    return directory


def lj_running_mean(NN, directory='/tmp/test-lj-running-mean-model'):
    graph = htf.graph_builder(NN)
    # pairwise energy. Double count -> divide by 2
    inv_r6 = graph.nlist_rinv**6
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    # sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    avg_energy = graph.running_mean(tf.reduce_sum(energy, axis=0),
                                    'average-energy')
    graph.save(force_tensor=forces, model_directory=directory,
               out_nodes=[avg_energy])
    return directory


def lj_force_output(NN, directory='/tmp/test-lj-rdf-model'):
    ops = []
    graph = htf.graph_builder(NN, output_forces=False)
    # pairwise energy. Double count -> divide by 2
    inv_r6 = graph.nlist_rinv**6
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    # sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    tf_forces = graph.compute_forces(energy)
    h_forces = graph.forces
    error = tf.losses.mean_squared_error(tf_forces, h_forces)
    v = tf.get_variable('error', shape=[])
    ops.append(v.assign(error))
    graph.save(model_directory=directory, out_nodes=ops)
    return directory


def lj_rdf(NN, directory='/tmp/test-lj-rdf-model'):
    graph = htf.graph_builder(NN)
    # pairwise energy. Double count -> divide by 2
    inv_r6 = graph.nlist_rinv**6
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    # sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1, name='energy')
    forces = graph.compute_forces(energy)
    # compute rdf between type 0 and 0
    rdf = graph.compute_rdf([3, 5], 'rdf', 10, 0, 0)
    avg_rdf = graph.running_mean(rdf, 'avg-rdf')
    # check = tf.add_check_numerics_ops()
    graph.save(force_tensor=forces, model_directory=directory)
    return directory


def lj_mol(NN, MN, directory='/tmp/test-lj-mol'):
    graph = htf.graph_builder(NN)
    graph.build_mol_rep(MN)
    # assume particle (w) is 0
    r = graph.safe_norm(graph.mol_nlist, axis=3)
    rinv = graph.safe_div(1.0, r)
    mol_p_energy = 4.0 / 2.0 * (rinv**12 - rinv**6)
    total_e = tf.reduce_sum(mol_p_energy)
    forces = graph.compute_forces(total_e)
    graph.save(force_tensor=forces, model_directory=directory, out_nodes=[])
    return directory


def print_graph(NN, directory='/tmp/test-print-model'):
    graph = htf.graph_builder(NN)
    nlist = graph.nlist[:, :, :3]
    # get r
    r = tf.norm(nlist, axis=2)
    # compute 1 / r while safely treating r = 0.
    # pairwise energy. Double count -> divide by 2
    inv_r6 = graph.safe_div(1., r**6)
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    # sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    prints = tf.Print(energy, [energy], summarize=1000)
    graph.save(force_tensor=forces, model_directory=directory,
               out_nodes=[prints])
    return directory


def trainable_graph(NN, directory='/tmp/test-trainable-model'):
    graph = htf.graph_builder(NN)
    nlist = graph.nlist[:, :, :3]
    # get r
    r = tf.norm(nlist, axis=2)
    # compute 1 / r while safely treating r = 0.
    # pairwise energy. Double count -> divide by 2
    epsilon = tf.Variable(1.0, name='lj-epsilon')
    sigma = tf.Variable(1.0, name='lj-sigma')
    tf.summary.scalar('lj-epsilon', epsilon)
    inv_r6 = graph.safe_div(sigma**6, r**6)
    p_energy = epsilon / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    # sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    check = tf.check_numerics(p_energy, 'Your tensor is invalid')
    forces = graph.compute_forces(energy)
    tf.summary.histogram('forces', forces)
    optimizer = tf.train.AdamOptimizer(1e-4)
    gvs = optimizer.compute_gradients(energy)
    print(gvs)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
    # for grad, var in gvs]
    train_op = optimizer.apply_gradients(gvs)
    # put non-trainable items
    # need to do reduction so batch size independent
    avg_energy = graph.running_mean(tf.reduce_sum(energy), 'avg-energy')
    # check = tf.add_check_numerics_ops()
    graph.save(force_tensor=forces, model_directory=directory,
               out_nodes=[train_op, check, avg_energy])
    return directory


def bootstrap_graph(NN, directory='/tmp/test-trainable-model'):
    # make bootstrap graph
    tf.reset_default_graph()
    v = tf.Variable(8.0, name='epsilon')
    s = tf.Variable(2.0, name='sigma')
    # save it
    bootstrap_dir = os.path.join(directory, 'bootstrap')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(bootstrap_dir, 'model'))
    return bootstrap_dir
