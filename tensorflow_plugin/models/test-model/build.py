import tensorflow as tf
import os


graph = hoomd.tensorflow_plugin.GraphBuilder(9, 9 - 1)
with tf.name_scope('force-calc') as scope:
    nlist = graph.nlist[:, :, :3]
    neighs_rs = tf.norm(nlist, axis=1, keepdims=True)
    #no need to use netwon's law because nlist should be double counted
    fr = tf.multiply(-1.0, tf.multiply(tf.reciprocal(neighs_rs), nlist), name='nan-pairwise-forces')
    with tf.name_scope('remove-nans') as scope:
        zeros = tf.zeros(tf.shape(nlist))
        real_fr = tf.where(tf.is_nan(fr), zeros, fr, name='pairwise-forces')
    forces = tf.reduce_sum(real_fr, axis=1, name='forces')
graph.save(forces, '/tmp/model')