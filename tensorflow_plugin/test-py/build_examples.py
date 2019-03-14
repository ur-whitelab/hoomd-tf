# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import os, hoomd.tensorflow_plugin, pickle


def simple_potential(directory='/tmp/test-simple-potential-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(9 - 1)
    with tf.name_scope('force-calc') as scope:
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(nlist, axis=2, keepdims=True)
        #no need to use netwon's law because nlist should be double counted
        fr = tf.multiply(-1.0, tf.multiply(tf.reciprocal(neighs_rs), nlist), name='nan-pairwise-forces')
        with tf.name_scope('remove-nans') as scope:
            zeros = tf.zeros_like(nlist)
            real_fr = tf.where(tf.is_nan(fr), zeros, fr, name='pairwise-forces')
        forces = tf.reduce_sum(real_fr, axis=1, name='forces')
    graph.save(force_tensor=forces, model_directory=directory)
    return directory

    #check graph info
    with open('/tmp/test-simple-potential-model/graph_info.p', 'rb') as f:
        gi = pickle.load(f)
        assert gi['forces'] != 'forces:0'
        assert tf.get_default_graph().get_tensor_by_name(gi['forces']).shape[1] == 4

def benchmark_gradient_potential():
    graph = hoomd.tensorflow_plugin.graph_builder(1024, 64)
    nlist = graph.nlist[:, :, :3]
    #get r
    r = tf.norm(nlist, axis=2)
    #compute 1 / r while safely treating r = 0.
    energy = tf.reduce_sum(graph.safe_div(1., r), axis=1)
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory='/tmp/benchmark-gradient-potential-model')

def gradient_potential():
    graph = hoomd.tensorflow_plugin.graph_builder(9 - 1)
    with tf.name_scope('force-calc') as scope:
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(nlist, axis=2)
        energy = 0.5 * graph.safe_div(numerator=tf.ones_like(neighs_rs, dtype=neighs_rs.dtype), denominator=neighs_rs, name='energy')
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory='/tmp/test-gradient-potential-model', out_nodes=[energy])

def noforce_graph(directory='/tmp/test-noforce-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(9 - 1, output_forces=False)
    nlist = graph.nlist[:, :, :3]
    neighs_rs = tf.norm(nlist, axis=2)
    energy = graph.safe_div(numerator=tf.ones_like(neighs_rs, dtype=neighs_rs.dtype), denominator=neighs_rs, name='energy')
    pos_norm = tf.norm(graph.positions, axis=1)
    graph.save(directory, out_nodes=[energy, pos_norm])
    return directory

def feeddict_graph(directory='/tmp/test-feeddict-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(9 - 1, output_forces=False)
    forces = graph.forces[:, :3]
    force_com = tf.reduce_mean(forces, axis=0)
    thing = tf.placeholder(dtype=tf.float32, name='test-tensor')
    out = force_com * thing
    graph.save(directory, out_nodes=[out])
    return directory

def benchmark_nonlist_graph(directory='/tmp/benchmark-nonlist-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(0, output_forces=True)
    ps = tf.norm(graph.positions, axis=1)
    energy = graph.safe_div(1. , ps)
    force = graph.compute_forces(energy)
    graph.save(directory, force_tensor=force, out_nodes=[energy])
    return directory

def lj_graph(NN, directory='/tmp/test-lj-potential-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
    nlist = graph.nlist[:, :, :3]
    #get r
    r = tf.norm(nlist, axis=2)
    #compute 1 / r while safely treating r = 0.
    #pairwise energy. Double count -> divide by 2
    inv_r6 = graph.safe_div(1., r**6)
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    #sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory=directory)
    return directory

def lj_running_mean(NN, directory='/tmp/test-lj-running-mean-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
    #pairwise energy. Double count -> divide by 2
    inv_r6 = graph.nlist_rinv**6
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    #sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    avg_energy = graph.running_mean(energy, 'average-energy')
    graph.save(force_tensor=forces, model_directory=directory, out_nodes=[avg_energy])
    return directory

def lj_rdf(NN, directory='/tmp/test-lj-rdf-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
    #pairwise energy. Double count -> divide by 2
    inv_r6 = graph.nlist_rinv**6
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    #sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    #compute rdf between type 0 and 0
    rdf = graph.compute_rdf([3, 5], 'rdf', 10, 0, 0)
    avg_rdf = graph.running_mean(rdf, 'avg-rdf')
    graph.save(force_tensor=forces, model_directory=directory)
    return directory

def print_graph(NN, directory='/tmp/test-print-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
    nlist = graph.nlist[:, :, :3]
    #get r
    r = tf.norm(nlist, axis=2)
    #compute 1 / r while safely treating r = 0.
    #pairwise energy. Double count -> divide by 2
    inv_r6 = graph.safe_div(1., r**6)
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    #sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    prints = tf.Print(energy, [energy], summarize=1000)
    graph.save(force_tensor=forces, model_directory=directory, out_nodes=[prints])
    return directory

def trainable_graph(NN, directory='/tmp/test-trainable-model'):
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
    nlist = graph.nlist[:, :, :3]
    #get r
    r = tf.norm(nlist, axis=2)
    #compute 1 / r while safely treating r = 0.
    #pairwise energy. Double count -> divide by 2
    epsilon = tf.Variable(1.0, name='lj-epsilon')
    sigma = tf.Variable(1.0, name='lj-sigma')
    tf.summary.scalar('lj-epsilon', epsilon)
    inv_r6 = graph.safe_div(sigma**6, r**6)
    p_energy = epsilon / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    #sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    check = tf.check_numerics(p_energy, 'Your tensor is invalid')
    forces = graph.compute_forces(energy)
    tf.summary.histogram('forces', forces)
    optimizer = tf.train.AdamOptimizer(1e-4)
    gvs = optimizer.compute_gradients(energy)
    capped_gvs = [(tf.clip_by_value(gvs, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
    #check = tf.add_check_numerics_ops()
    graph.save(force_tensor=forces, model_directory=directory, out_nodes=[train_op, check])
    return directory

def bootstrap_graph(NN, directory='/tmp/test-trainable-model'):
    #make bootstrap graph
    tf.reset_default_graph()
    v = tf.Variable(8.0, name='epsilon')
    s = tf.Variable(2.0, name='sigma')

    #save it
    bootstrap_dir = os.path.join(directory, 'bootstrap')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(bootstrap_dir, 'model'))
    return bootstrap_dir

