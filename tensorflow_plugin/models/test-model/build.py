import tensorflow as tf
import os


N = 3 * 3
NN = N - 1
save_loc = '/tmp'

nlist = tf.ones ([N * NN, 4], name='nlist')
_ = tf.ones([N, 4], name='positions')

with tf.name_scope('force-calc') as scope:
    neighs = tf.reshape(nlist, [N, NN, 4], name='reshaped-nlist')
    neighs_rs = tf.norm(neighs, axis=1, keepdims=True)
    normed_neighs = tf.divide(neighs, neighs_rs, name='normed-neighs')
    #1/2 because we're double counting
    fr = tf.multiply(-0.5, tf.multiply(tf.reciprocal(neighs_rs), normed_neighs), name='nan-pairwise-forces')
    with tf.name_scope('remove-nans') as scope:
        zeros = tf.zeros(tf.shape(neighs))
        real_fr = tf.where(tf.is_nan(fr), zeros, fr, name='pairwise-forces')
forces = tf.reduce_sum(real_fr, axis=1, name='forces')
#need to make variable so we have something to save.
#confusing, I know
tf.Variable(forces, name='force-save')

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(save_loc, 'model'))