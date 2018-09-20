import tensorflow as tf
import os, hoomd.tensorflow_plugin, pickle


def simple_potential():
    graph = hoomd.tensorflow_plugin.graph_builder(9, 9 - 1)
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
    graph = hoomd.tensorflow_plugin.graph_builder(1024, 64)
    nlist = graph.nlist[:, :, :3]
    #get r
    r = tf.norm(nlist, axis=2)
    #compute 1 / r while safely treating r = 0.
    energy = tf.reduce_sum(graph.safe_div(1., r), axis=1)
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory='/tmp/benchmark-gradient-potential-model')

def gradient_potential():
    graph = hoomd.tensorflow_plugin.graph_builder(9, 9 - 1)
    with tf.name_scope('force-calc') as scope:
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(nlist, axis=2)
        energy = 0.5 * graph.safe_div(numerator=tf.ones(neighs_rs.shape, dtype=neighs_rs.dtype), denominator=neighs_rs, name='energy')
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory='/tmp/test-gradient-potential-model', out_nodes=[energy])

def noforce_graph():
    graph = hoomd.tensorflow_plugin.graph_builder(9, 9 - 1, output_forces=False)
    nlist = graph.nlist[:, :, :3]
    neighs_rs = tf.norm(nlist, axis=2)
    energy = graph.safe_div(numerator=tf.ones(neighs_rs.shape, dtype=neighs_rs.dtype), denominator=neighs_rs, name='energy')
    graph.save('/tmp/test-noforce-model', out_nodes=[energy])

def feeddict_graph():
    graph = hoomd.tensorflow_plugin.graph_builder(9, 9 - 1, output_forces=False)
    forces = graph.forces[:, :3]
    force_com = tf.reduce_mean(forces, axis=0)
    thing = tf.placeholder(dtype=tf.float32, name='test-tensor')
    out = force_com * thing
    graph.save('/tmp/test-feeddict-model', out_nodes=[out])

def benchmark_nonlist_graph():
    graph = hoomd.tensorflow_plugin.graph_builder(1024, 0, output_forces=False)
    ps = tf.norm(graph.positions, axis=1)
    energy = graph.safe_div(1. , ps)
    graph.save('/tmp/benchmark-nonlist-model', out_nodes=[energy])

def lj_graph(N, NN, name):
    graph = hoomd.tensorflow_plugin.graph_builder(N, NN)
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
    graph.save(force_tensor=forces, model_directory=name)

def print_graph(N, NN, name):
    graph = hoomd.tensorflow_plugin.graph_builder(N, NN)
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
    graph.save(force_tensor=forces, model_directory=name, out_nodes=[prints])

def trainable_graph(N, NN, name):
    graph = hoomd.tensorflow_plugin.graph_builder(N, NN)
    nlist = graph.nlist[:, :, :3]
    #get r
    r = tf.norm(nlist, axis=2)
    #compute 1 / r while safely treating r = 0.
    #pairwise energy. Double count -> divide by 2
    inv_r6 = graph.safe_div(1., r**6)
    epsilon = tf.Variable(4.0, name='lj-epsilon')
    tf.summary.scalar('lj-epsilon', epsilon)
    p_energy = epsilon / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    #sum over pairwise energy
    energy = tf.reduce_sum(p_energy, axis=1)
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces, model_directory=name)

feeddict_graph()
noforce_graph()
gradient_potential()
simple_potential()
benchmark_gradient_potential()
benchmark_nonlist_graph()
lj_graph(2**14, 64, '/tmp/benchmark-lj-potential-model')
lj_graph(9, 9 - 1, '/tmp/test-lj-potential-model')
print_graph(9, 9 - 1, '/tmp/test-print-model')
trainable_graph(9, 9 - 1, '/tmp/test-trainable-model')