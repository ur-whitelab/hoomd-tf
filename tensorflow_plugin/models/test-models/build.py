import tensorflow as tf
import os, hoomd.tensorflow_plugin, pickle


def simple_potential():
    graph = hoomd.tensorflow_plugin.GraphBuilder(9, 9 - 1)
    with tf.name_scope('force-calc') as scope:
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(nlist, axis=2, keepdims=True)
        #no need to use netwon's law because nlist should be double counted
        fr = tf.multiply(-1.0, tf.multiply(tf.reciprocal(neighs_rs), nlist), name='nan-pairwise-forces')
        with tf.name_scope('remove-nans') as scope:
            zeros = tf.zeros(tf.shape(nlist))
            real_fr = tf.where(tf.is_nan(fr), zeros, fr, name='pairwise-forces')
        forces = tf.reduce_sum(real_fr, axis=1, name='forces')
    graph.save(force_tensor=forces, model_directory='/tmp/test-simple-potential-model')

    #check graph info
    with open('/tmp/test-simple-potential-model/graph_info.p', 'rb') as f:
        gi = pickle.load(f)
        assert gi['forces'] != 'forces:0'
        assert tf.get_default_graph().get_tensor_by_name(gi['forces']).shape[1] == 4

def benchmark_gradient_potential():
    graph = hoomd.tensorflow_plugin.GraphBuilder(1024, 128)
    nlist = graph.nlist[:, :, :3]
    #get r
    print(nlist.shape)
    r = tf.norm(nlist, axis=2)
    #compute 1 / r while safely treating r = 0.
    energy = tf.reduce_sum(graph.safe_div(1., r), axis=1)
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory='/tmp/benchmark-gradient-potential-model')

def gradient_potential():
    graph = hoomd.tensorflow_plugin.GraphBuilder(9, 9 - 1)
    with tf.name_scope('force-calc') as scope:
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(nlist, axis=2)
        energy = graph.safe_div(numerator=tf.ones(neighs_rs.shape, dtype=neighs_rs.dtype), denominator=neighs_rs, name='energy')
    forces = graph.compute_forces(energy)
    print(forces, forces.shape)
    graph.save(force_tensor=forces, model_directory='/tmp/test-gradient-potential-model', out_nodes=[energy])

def noforce_graph():
    graph = hoomd.tensorflow_plugin.GraphBuilder(9, 9 - 1, output_forces=False)
    nlist = graph.nlist[:, :, :3]
    neighs_rs = tf.norm(nlist, axis=2)
    energy = graph.safe_div(numerator=tf.ones(neighs_rs.shape, dtype=neighs_rs.dtype), denominator=neighs_rs, name='energy')
    graph.save('/tmp/test-noforce-model', out_nodes=[energy])

def benchmark_nonlist_graph():
    graph = hoomd.tensorflow_plugin.GraphBuilder(1024, 0, output_forces=False)
    ps = tf.norm(graph.positions, axis=1)
    energy = tf.reduce_sum(graph.safe_div(1. , ps), axis=1)
    graph.save('/tmp/benchmark-nonlist-model', out_nodes=[energy])

noforce_graph()
gradient_potential()
simple_potential()
benchmark_gradient_potential()
benchmark_nonlist_graph()